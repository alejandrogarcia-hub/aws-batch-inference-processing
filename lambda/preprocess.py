"""Preprocessing Lambda function for Bedrock batch inference jobs.

This module handles the preparation of input data for Amazon Bedrock batch inference.
It transforms user-provided datasets (CSV/Parquet from S3 or Hugging Face) into properly
formatted JSONL files that conform to Bedrock's batch API requirements.

Key responsibilities:
- Load and validate input datasets from various sources
- Apply prompt templates to format text for language models
- Chunk large datasets to respect Bedrock's job size limits
- Generate unique record IDs for output correlation
- Create job configurations for parallel processing

"""

import json
import os
from collections.abc import Mapping
from string import Formatter
from typing import Any
from uuid import uuid4

import awswrangler as wr
import boto3
import pandas as pd
import prompt_templates as pt
import utils
from custom_types import JobConfig, JobConfigList, JobInput
from datasets import load_dataset
from processor import get_processor_for_model_id


def _parse_optional_int(value: object | None) -> int | None:
    """Parse an optional integer value from various input types.

    Handles None, empty strings, and the literal string 'none' as None values.
    This flexibility supports both programmatic and user-provided inputs.

    Args:
        value: Input value that may be None, string, or numeric.
            Accepts: None, "", "none", "None", numeric strings, or numbers.

    Returns:
        Parsed integer value or None if input represents absence of value.

    Raises:
        ValueError: If the value cannot be parsed as an integer.

    Examples:
        >>> _parse_optional_int(None)
        None
        >>> _parse_optional_int("100")
        100
        >>> _parse_optional_int("")
        None
        >>> _parse_optional_int("none")
        None
    """
    if value is None:
        return None

    # Handle string inputs with various representations of None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped.lower() == "none":
            return None
        value = stripped

    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected an integer value.") from exc

    return parsed


# Configuration constants from environment variables
# These control batch processing behavior and resource allocation

# Maximum number of records to include in a single Bedrock batch job
# Bedrock has limits on job size (typically 50,000 records max)
MAX_RECORDS_PER_JOB: int = int(os.getenv("MAX_RECORDS_PER_JOB", "1000"))

# Optional global limit on total records to process across all jobs
# Useful for testing or cost control
_max_total_records_env = os.getenv("MAX_TOTAL_RECORDS")
MAX_TOTAL_RECORDS: int | None = (
    None
    if _max_total_records_env is None or _max_total_records_env.strip() == ""
    else int(_max_total_records_env)
)

# S3 bucket for storing preprocessed JSONL files and intermediate data
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Initialize AWS service clients
s3_client = boto3.client("s3")

# Configure structured logging for CloudWatch
logger = utils.get_logger()


def resolve_prompt_configuration(
    model_type: str, event: JobInput | Mapping[str, Any]
) -> tuple[str | None, set[str]]:
    """Resolve and validate prompt template configuration for text generation models.

    This function determines which prompt template to use based on the job configuration
    and extracts all placeholder fields that must be present in the input data.
    For embedding models, no prompt template is needed, so empty values are returned.

    Args:
        model_type: Processor model type ('text' for LLMs, 'embedding' for vector models).
        event: Incoming job configuration containing prompt_id and other parameters.
            Can be a JobInput instance or a dictionary-like mapping.

    Returns:
        A tuple containing:
            - prompt_template: The formatted string template with {placeholders}, or None for embeddings.
            - required_fields: Set of field names that must exist in the input data columns.

    Raises:
        ValueError: If a text model job is missing prompt_id or references an unknown template.

    Examples:
        For a template like "Tell me about {topic} in {style} style":
        >>> resolve_prompt_configuration("text", {"prompt_id": "explain_topic"})
        ("Tell me about {topic} in {style} style", {"topic", "style"})

        For embedding models:
        >>> resolve_prompt_configuration("embedding", {})
        (None, set())
    """
    # Normalize input to JobInput type for consistent access
    job_input = event if isinstance(event, JobInput) else JobInput.safe_validate(event)

    # Embedding models don't use prompt templates - they process raw text
    if model_type != "text":
        return None, set()

    # Text models require a prompt_id to identify which template to use
    prompt_id = job_input.prompt_id
    if not prompt_id:
        raise ValueError(
            "Text models require a prompt_id that matches an entry in prompt_templates.prompt_id_to_template."
        )

    # Look up the prompt template from the registered templates
    try:
        prompt_template = pt.prompt_id_to_template[prompt_id]
    except KeyError as exc:
        # Provide helpful error message listing available options
        available = ", ".join(sorted(pt.prompt_id_to_template)) or "<none>"
        raise ValueError(
            f'Unknown prompt_id "{prompt_id}". Available prompt_ids: {available}.'
        ) from exc

    # Parse the template to extract all placeholder field names
    # Formatter.parse returns tuples of (literal_text, field_name, format_spec, conversion)
    formatter = Formatter()
    required_fields = {
        field for _, field, _, _ in formatter.parse(prompt_template) if field
    }

    return prompt_template, required_fields


def validate_chunk_columns(
    model_type: str,
    columns: set[str],
    *,
    required_prompt_fields: set[str],
    job_name_prefix: str,
    prompt_id: str | None,
) -> None:
    """Validate that a data chunk contains all required columns for the model type.

    This validation ensures that:
    - Embedding models have an 'input_text' column for raw text
    - Text generation models have all columns referenced in their prompt template

    The validation is performed per chunk to catch data issues early and provide
    clear error messages before attempting to process the data.

    Args:
        model_type: Type of processor ('embedding' or 'text').
        columns: Set of column names present in the current dataframe chunk.
        required_prompt_fields: Field names extracted from the prompt template.
            Empty set for embedding models.
        job_name_prefix: User-provided job identifier for error context.
        prompt_id: Template identifier for text models, None for embeddings.

    Raises:
        ValueError: If required columns are missing, with details about which
            columns are needed and which prompt template requires them.

    Note:
        The * in the signature enforces keyword-only arguments after columns,
        improving API clarity and preventing positional argument errors.
    """
    if model_type == "embedding":
        # Embedding models require a standardized 'input_text' column
        if "input_text" not in columns:
            raise ValueError(
                'Embedding jobs require an "input_text" column in the input dataset. '
                f"Job prefix: {job_name_prefix}"
            )
        return

    # For text models, verify all template placeholders have corresponding columns
    missing = required_prompt_fields - columns
    if missing:
        raise ValueError(
            "Input dataset is missing the following columns required by "
            f'prompt_id "{prompt_id}": {sorted(missing)}'
        )


def write_jsonl_to_s3(records: list[dict], key: str) -> str:
    """Write a list of dictionaries as a JSONL file to S3.

    JSONL (JSON Lines) format has one JSON object per line, which is required
    by Bedrock's batch inference API. This format allows for streaming processing
    of large datasets.

    Args:
        records: List of dictionaries to serialize. Each dict becomes one line.
            Typically contains 'recordId' and 'modelInput' keys for Bedrock.
        key: S3 object key (path) where the JSONL file will be stored.

    Returns:
        Complete S3 URI of the uploaded file (s3://bucket/key format).

    Note:
        - Each record is serialized with json.dumps() ensuring valid JSON
        - No compression is applied as Bedrock expects raw JSONL
        - The entire file is built in memory, so very large datasets may
          require streaming approaches
    """
    # Create JSONL format: one JSON object per line
    jsonl_data = "\n".join(json.dumps(item) for item in records)

    # Upload to S3
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=jsonl_data)

    return f"s3://{BUCKET_NAME}/{key}"


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda handler for preprocessing datasets for Bedrock batch inference.

    This function orchestrates the entire preprocessing workflow:
    1. Validates input parameters and resolves data source
    2. Loads data from S3 or Hugging Face in chunks
    3. Applies prompt templates (for text models) or validates input format (for embeddings)
    4. Splits data into appropriately sized batches for Bedrock
    5. Generates JSONL files conforming to Bedrock's API requirements
    6. Returns job configurations for Step Functions to process in parallel

    Event structure is a JobInput dictionary, examples:

    For Titan-V2 embedding jobs
    {
      "s3_uri": "s3://batch-inference-bucket-xxxxxxxxx/inputs/embeddings/embedding_input.csv",
      "job_name_prefix": "test-embeddings-job1",
      "model_id": "amazon.titan-embed-text-v2:0",
      "prompt_id": null
    }
    The s3_uri must point to a CSV file with an `input_text` column for embedding models.

    For text-based models like the Anthropic Claude family, you must supply a value for prompt_id that is associated
    with a prompt template in `prompt_templates.prompt_id_to_template`.
    The input CSV must have columns for each formatting key in the prompt template.
    {
      "s3_uri": "s3://batch-inference-bucket-xxxxxxxxx/inputs/jokes/topics.csv",
      "job_name_prefix": "test-joke-job1",
      "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
      "prompt_id": "joke_about_topic"
    }

    Returns a list of job configs which will be passed to the start_batch_inference_job.py function via a step function
    map, which manages concurrency of the requests.
    Args:
        event: Input event containing job configuration. Must conform to JobInput schema.
        context: Lambda context object (contains request ID, function name, etc.).
            Not used directly but required by Lambda runtime.

    Returns:
        Dictionary containing 'jobs' key with list of JobConfig objects.
        Each JobConfig specifies one Bedrock batch inference job to execute.

    Raises:
        ValueError: If input validation fails (missing required fields, invalid values).
        KeyError: If specified prompt_id doesn't exist in templates.
    """
    # Validate and parse the incoming event using Pydantic model
    job_input = JobInput.model_validate(event)

    # Ensure we have a data source (either S3 or Hugging Face, but not both)
    if job_input.dataset_id is None and job_input.s3_uri is None:
        raise ValueError(
            "Either 'dataset_id' or 's3_uri' must be provided in the event."
        )

    # Determine the appropriate processor based on model type
    model_id = job_input.model_id
    processor = get_processor_for_model_id(model_id)

    # Resolve prompt template configuration for text models
    # For embeddings, this returns (None, empty_set)
    prompt_template, required_prompt_fields = resolve_prompt_configuration(
        processor.model_type, job_input
    )

    # Extract and validate job processing limits
    max_num_jobs = job_input.max_num_jobs  # Max number of parallel jobs to create
    max_records_per_job = job_input.max_records_per_job or MAX_RECORDS_PER_JOB

    # Handle optional total record limit (useful for testing/cost control)
    max_records_total_raw = (
        job_input.max_records_total
        if job_input.max_records_total is not None
        else MAX_TOTAL_RECORDS
    )
    max_records_total = _parse_optional_int(max_records_total_raw)
    if max_records_total is not None and max_records_total <= 0:
        raise ValueError("max_records_total must be a positive integer when provided.")

    # Branch 1: Load data from Hugging Face datasets
    if dataset_id := job_input.dataset_id:  # Walrus operator for assignment + check
        logger.info(f"Writing huggingface dataset {dataset_id} to S3")

        # Stage Hugging Face data to S3 for processing
        s3_uri = f"s3://{BUCKET_NAME}/hf/{dataset_id}"
        file_type = "parquet"  # HF datasets are converted to Parquet for efficiency

        # Stream dataset in batches to avoid loading entire dataset into memory
        batched_ds = load_dataset(
            dataset_id, split=job_input.split, streaming=True
        ).batch(batch_size=max_records_per_job)

        for idx, batch in enumerate(batched_ds):
            df = pd.DataFrame(batch)
            # Write each batch as a separate Parquet file with zero-padded naming
            wr.s3.to_parquet(
                df,
                path=f"{s3_uri}/{str(idx).zfill(4)}.snappy.parquet",
                index=False,
                compression="snappy",  # Snappy compression for faster reads
            )

            # Early exit if we've created enough jobs
            if max_num_jobs and idx >= max_num_jobs:
                break
    else:
        # Branch 2: Load data directly from S3
        if job_input.s3_uri is None:
            raise ValueError("Missing 's3_uri' for preprocessing input.")

        s3_uri = job_input.s3_uri
        # Determine file type from extension
        file_type = s3_uri.split(".")[-1]
        assert file_type in ["csv", "parquet"], "File type must be csv or parquet"
        logger.info(f"Using S3 dataset at {s3_uri}")

    # Process input data in chunks to create batch inference jobs
    jobs: list[JobConfig] = []  # Accumulator for job configurations

    logger.info("Preparing batch inference job inputs (JSONL files)...")
    processed_records = 0  # Track total records for limit enforcement

    # Iterate through data chunks, each becoming a separate batch job
    for idx, input_df in utils.load_files_in_chunks(
        s3_uri, file_type, chunk_size=max_records_per_job
    ):
        # Check if we've created enough jobs (useful for testing)
        if max_num_jobs and idx >= max_num_jobs:
            logger.info(f"Reached max_num_jobs: {max_num_jobs}. Stopping here.")
            break

        # Check if we've processed enough total records
        if max_records_total is not None and processed_records >= max_records_total:
            logger.info("Reached max_records_total limit. Stopping here.")
            break

        # Generate unique record IDs for output correlation
        # These IDs allow us to match Bedrock outputs back to original inputs
        if "record_id" not in input_df.columns:
            input_df["record_id"] = [str(uuid4()) for _ in range(len(input_df))]

        # Handle partial chunk if it would exceed total record limit
        if (
            max_records_total is not None
            and processed_records + len(input_df) > max_records_total
        ):
            allowed = max_records_total - processed_records
            logger.info(
                f"Truncating chunk to {allowed} records to respect max_records_total {max_records_total}."
            )
            input_df = input_df.head(allowed)
            if len(input_df) == 0:
                break

        # Convert DataFrame to list of dictionaries for processing
        input_records = input_df.to_dict("records")

        # Validate that all required columns are present
        validate_chunk_columns(
            processor.model_type,
            set(input_df.columns),
            required_prompt_fields=required_prompt_fields,
            job_name_prefix=job_input.job_name_prefix,
            prompt_id=job_input.prompt_id,
        )

        # Transform records into Bedrock API format
        if processor.model_type == "embedding":
            # Embedding models: pass raw text directly
            records = [
                processor.process_input(
                    input_text=r["input_text"], record_id=r["record_id"]
                )
                for r in input_records
            ]
        else:
            # Text models: apply prompt template to format input
            # The format() call replaces {placeholders} with column values
            records = [
                processor.process_input(
                    input_text=prompt_template.format(**r), record_id=r["record_id"]
                )
                for r in input_records
            ]

        # Generate unique job name for this chunk
        # Job names must be unique across all Bedrock batch jobs in the account
        job_name = utils.create_job_name(job_input.job_name_prefix, index=idx)

        # Construct S3 paths for this chunk's data
        # We maintain a consistent naming convention with zero-padded indices
        # for easy sorting and identification of chunk order
        input_parquet_path = f"s3://{BUCKET_NAME}/batch_inputs_parquet/{job_input.job_name_prefix}/{str(idx).zfill(4)}.snappy.parquet"
        input_key = (
            f"batch_inputs_json/{job_input.job_name_prefix}/{str(idx).zfill(4)}.jsonl"
        )
        output_path = f"s3://{BUCKET_NAME}/batch_outputs_json/{job_input.job_name_prefix}/{str(idx).zfill(4)}/"

        # Clean up any unnamed index columns that pandas might have added
        # These are typically created when reading CSV files with unnamed first columns
        if "Unnamed: 0" in input_df.columns:
            input_df = input_df.drop(columns=["Unnamed: 0"])

        # Save the original input data as Parquet for later joining with outputs
        # This preserves the original data alongside record_ids for correlation
        # Snappy compression provides a good balance of speed and size reduction
        wr.s3.to_parquet(
            input_df, path=input_parquet_path, index=False, compression="snappy"
        )

        # Create job configuration for this chunk
        # This config will be passed to start_batch_inference_job Lambda
        job_config = JobConfig(
            model_id=model_id,
            job_name=job_name,
            input_parquet_path=input_parquet_path,  # Original data for postprocessing joins
            s3_uri_input=write_jsonl_to_s3(records, input_key),  # Formatted JSONL for Bedrock
            s3_uri_output=output_path,  # Directory where Bedrock will write results
        )
        jobs.append(job_config)

        # Update running total and check if we should stop processing
        processed_records += len(input_df)
        if max_records_total is not None and processed_records >= max_records_total:
            logger.info(
                "Reached max_records_total limit after current chunk. Stopping."
            )
            break

    # Package all job configurations for the Step Functions Map state
    # The Map state will process these jobs in parallel up to the concurrency limit
    job_config_list = JobConfigList(jobs=jobs)

    # Return the job list as a dictionary, excluding None values for cleaner JSON
    # This output becomes the input for the Step Functions Map state
    return job_config_list.model_dump(exclude_none=True)
