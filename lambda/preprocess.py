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
    if value is None:
        return None
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


MAX_RECORDS_PER_JOB: int = int(os.getenv("MAX_RECORDS_PER_JOB", "1000"))
_max_total_records_env = os.getenv("MAX_TOTAL_RECORDS")
MAX_TOTAL_RECORDS: int | None = (
    None
    if _max_total_records_env is None or _max_total_records_env.strip() == ""
    else int(_max_total_records_env)
)
BUCKET_NAME = os.getenv("BUCKET_NAME")

s3_client = boto3.client("s3")

logger = utils.get_logger()


def resolve_prompt_configuration(
    model_type: str, event: JobInput | Mapping[str, Any]
) -> tuple[str | None, set[str]]:
    """Return the prompt template and required fields for text models.

    Args:
        model_type: Processor model type (for example ``text`` or ``embedding``).
        event: Incoming job configuration payload.

    Returns:
        Tuple of (prompt_template, required_fields). For non-text models both values are empty.

    Raises:
        ValueError: When text models are missing ``prompt_id`` or reference an unknown template.
    """

    job_input = event if isinstance(event, JobInput) else JobInput.safe_validate(event)

    if model_type != "text":
        return None, set()

    prompt_id = job_input.prompt_id
    if not prompt_id:
        raise ValueError(
            "Text models require a prompt_id that matches an entry in prompt_templates.prompt_id_to_template."
        )

    try:
        prompt_template = pt.prompt_id_to_template[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(pt.prompt_id_to_template)) or "<none>"
        raise ValueError(
            f'Unknown prompt_id "{prompt_id}". Available prompt_ids: {available}.'
        ) from exc

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
    """Ensure each dataframe chunk contains the columns needed for processing.

    Args:
        model_type: Processor type (``embedding`` or ``text``).
        columns: Set of column names present in the current dataframe chunk.
        required_prompt_fields: Keys referenced in the prompt template (empty for embeddings).
        job_name_prefix: User-provided prefix used for log context.
        prompt_id: Identifier for the prompt template; may be ``None`` for embeddings.

    Raises:
        ValueError: When the chunk is missing required columns.
    """

    if model_type == "embedding":
        if "input_text" not in columns:
            raise ValueError(
                'Embedding jobs require an "input_text" column in the input dataset. '
                f"Job prefix: {job_name_prefix}"
            )
        return

    missing = required_prompt_fields - columns
    if missing:
        raise ValueError(
            "Input dataset is missing the following columns required by "
            f'prompt_id "{prompt_id}": {sorted(missing)}'
        )


def write_jsonl_to_s3(records: list[dict], key: str) -> str:
    """write a JSONL file to S3 from a list of dicts. Returns the S3 URI"""
    jsonl_data = "\n".join(json.dumps(item) for item in records)
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=jsonl_data)
    return f"s3://{BUCKET_NAME}/{key}"


def lambda_handler(event: dict[str, Any], context) -> dict[str, Any]:
    """
    Preprocessing of input CSV files and preparation of JSONL batch input files for bedrock batch inference.

    Event structure is a JobInput TypedDict, e.g. for Titan-V2 embedding jobs
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
    """

    job_input = JobInput.model_validate(event)

    if job_input.dataset_id is None and job_input.s3_uri is None:
        raise ValueError(
            "Either 'dataset_id' or 's3_uri' must be provided in the event."
        )

    model_id = job_input.model_id
    processor = get_processor_for_model_id(model_id)

    prompt_template, required_prompt_fields = resolve_prompt_configuration(
        processor.model_type, job_input
    )

    max_num_jobs = job_input.max_num_jobs
    max_records_per_job = job_input.max_records_per_job or MAX_RECORDS_PER_JOB
    max_records_total_raw = (
        job_input.max_records_total
        if job_input.max_records_total is not None
        else MAX_TOTAL_RECORDS
    )
    max_records_total = _parse_optional_int(max_records_total_raw)
    if max_records_total is not None and max_records_total <= 0:
        raise ValueError("max_records_total must be a positive integer when provided.")

    # huggingface datasets - load and import to S3
    if dataset_id := job_input.dataset_id:
        logger.info(f"Writing huggingface dataset {dataset_id} to S3")

        s3_uri = f"s3://{BUCKET_NAME}/hf/{dataset_id}"
        file_type = "parquet"

        batched_ds = load_dataset(
            dataset_id, split=job_input.split, streaming=True
        ).batch(batch_size=max_records_per_job)
        for idx, batch in enumerate(batched_ds):
            df = pd.DataFrame(batch)
            wr.s3.to_parquet(
                df,
                path=f"{s3_uri}/{str(idx).zfill(4)}.snappy.parquet",
                index=False,
                compression="snappy",
            )

            if max_num_jobs:
                if idx >= max_num_jobs:
                    break
    else:
        # load directly from S3
        if job_input.s3_uri is None:
            raise ValueError("Missing 's3_uri' for preprocessing input.")
        s3_uri = job_input.s3_uri
        file_type = s3_uri.split(".")[-1]
        assert file_type in ["csv", "parquet"], "File type must be csv or parquet"
        logger.info(f"Using S3 dataset at {s3_uri}")

    # load input in chunks
    jobs: list[JobConfig] = []

    logger.info("Preparing batch inference job inputs (JSONL files)...")
    processed_records = 0
    for idx, input_df in utils.load_files_in_chunks(
        s3_uri, file_type, chunk_size=max_records_per_job
    ):
        if max_num_jobs:
            if idx >= max_num_jobs:
                logger.info(f"Reached max_num_jobs: {max_num_jobs}. Stopping here.")
                break

        if max_records_total is not None and processed_records >= max_records_total:
            logger.info("Reached max_records_total limit. Stopping here.")
            break

        # add a record_id to each row to allow for joining with outputs later
        if "record_id" not in input_df.columns:
            input_df["record_id"] = [str(uuid4()) for _ in range(len(input_df))]

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

        input_records = input_df.to_dict("records")
        validate_chunk_columns(
            processor.model_type,
            set(input_df.columns),
            required_prompt_fields=required_prompt_fields,
            job_name_prefix=job_input.job_name_prefix,
            prompt_id=job_input.prompt_id,
        )

        # transformation function
        if processor.model_type == "embedding":
            records = [
                processor.process_input(
                    input_text=r["input_text"], record_id=r["record_id"]
                )
                for r in input_records
            ]

        else:
            # format the prompt - input df must have columns that match the formatting keys in the prompt
            records = [
                processor.process_input(
                    input_text=prompt_template.format(**r), record_id=r["record_id"]
                )
                for r in input_records
            ]

        job_name = utils.create_job_name(job_input.job_name_prefix, index=idx)

        input_parquet_path = f"s3://{BUCKET_NAME}/batch_inputs_parquet/{job_input.job_name_prefix}/{str(idx).zfill(4)}.snappy.parquet"
        input_key = (
            f"batch_inputs_json/{job_input.job_name_prefix}/{str(idx).zfill(4)}.jsonl"
        )
        output_path = f"s3://{BUCKET_NAME}/batch_outputs_json/{job_input.job_name_prefix}/{str(idx).zfill(4)}/"

        if "Unnamed: 0" in input_df.columns:
            input_df = input_df.drop(columns=["Unnamed: 0"])
        # save this file and keep in the config to allow for joins to the output by record id
        wr.s3.to_parquet(
            input_df, path=input_parquet_path, index=False, compression="snappy"
        )

        job_config = JobConfig(
            model_id=model_id,
            job_name=job_name,
            input_parquet_path=input_parquet_path,
            s3_uri_input=write_jsonl_to_s3(records, input_key),
            s3_uri_output=output_path,
        )
        jobs.append(job_config)

        processed_records += len(input_df)
        if max_records_total is not None and processed_records >= max_records_total:
            logger.info(
                "Reached max_records_total limit after current chunk. Stopping."
            )
            break

    job_config_list = JobConfigList(jobs=jobs)

    return job_config_list.model_dump(exclude_none=True)
