"""Postprocessing Lambda function for Bedrock batch inference results.

This module handles the final stage of the batch inference pipeline, where
model outputs are parsed, joined with original input data, and saved in an
analysis-ready format. The postprocessor transforms Bedrock's JSONL output
into structured Parquet files with complete input-output pairs.

Key responsibilities:
- Parse JSONL output files from completed Bedrock batch jobs
- Extract and normalize model responses using processor classes
- Join outputs with original inputs using record IDs
- Save enriched datasets as Parquet files for downstream analysis
- Handle error cases gracefully by skipping failed jobs

The postprocessing step is critical for:
- Data lineage: Maintaining connection between inputs and outputs
- Format standardization: Converting from JSONL to columnar Parquet
- Error resilience: Processing successful jobs even if others fail
- Analytics readiness: Creating queryable datasets for analysis

Typical workflow:
1. Step Functions triggers this Lambda after batch job completion
2. Lambda retrieves the output JSONL from S3
3. Parses and processes each output record
4. Joins with original input data
5. Saves complete dataset as Parquet
"""

import json
import os
from collections.abc import Iterable

import awswrangler as wr
import boto3
import pandas as pd

import utils
from custom_types import TaskItem
from processor import get_processor_for_model_id

# Configure structured logging for CloudWatch
logger = utils.get_logger()

# S3 bucket for storing processed outputs
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Initialize AWS service client for S3 operations
s3_client = boto3.client("s3")


def _normalize_lines(lines: Iterable[str | bytes]) -> Iterable[str]:
    """Normalize line endings and encoding for consistent JSONL parsing.

    Handles both string and bytes input, ensuring all lines are properly
    decoded UTF-8 strings with trailing newlines removed. This normalization
    is essential for robust JSONL parsing across different data sources.

    Args:
        lines: Iterable of lines that may be strings or bytes.
            Typically from S3 object streaming or file reading.

    Yields:
        UTF-8 decoded strings with trailing newlines stripped.
        Each yielded line represents one JSON document.

    Note:
        This generator function processes lines lazily, making it
        memory-efficient for large JSONL files.
    """
    for line in lines:
        # Handle both bytes (from S3 streaming) and string inputs
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        # Remove trailing newlines while preserving line content
        yield line.rstrip("\n")


def parse_jsonl(
    lines: Iterable[str | bytes], *, context: str | None = None
) -> list[dict]:
    """Parse JSON Lines format into Python dictionaries with error handling.

    Robust JSONL parser that handles common edge cases like blank lines
    and provides detailed error messages for debugging failed parses.
    JSONL format has one JSON object per line, making it ideal for
    streaming large datasets.

    Args:
        lines: Iterable yielding each JSON document as a string or bytes.
            Each line should contain a complete JSON object.
        context: Optional identifier (e.g., S3 URI) included in error messages
            to help locate problematic files in multi-file processing.

    Returns:
        List of parsed JSON objects maintaining original order.
        Empty lines are skipped and not included in the output.

    Raises:
        ValueError: If any non-empty line contains invalid JSON.
            The error includes line number and content for debugging.

    Example:
        >>> lines = ['{"id": 1}', '', '{"id": 2}']
        >>> parse_jsonl(lines)
        [{'id': 1}, {'id': 2}]
    """
    records: list[dict] = []

    # Process each line with line number for error reporting
    for line_number, raw_line in enumerate(_normalize_lines(lines), start=1):
        line = raw_line.strip()

        # Skip empty lines gracefully - common in JSONL files
        if not line:
            continue

        try:
            # Parse JSON and add to results
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # Provide detailed error context for debugging
            location = f" {context}" if context else ""
            raise ValueError(
                f"Failed to parse JSONL{location} on line {line_number}: {line!r}"
            ) from exc

    return records


def read_jsonl_from_s3(s3_uri: str) -> list[dict]:
    """Fetch and parse a JSONL file from S3 storage.

    Efficiently streams JSONL content from S3 without loading the entire
    file into memory. This is important for handling large batch inference
    outputs that may contain thousands of records.

    Args:
        s3_uri: Full S3 URI in format 's3://bucket/path/to/file.jsonl'.
            Must point to a valid JSONL file with read permissions.

    Returns:
        List of dictionaries parsed from the JSONL file.
        Each dictionary represents one output record from Bedrock.

    Raises:
        ValueError: If S3 URI is malformed or JSONL parsing fails.
        ClientError: If S3 object doesn't exist or access is denied.

    Note:
        Uses streaming to handle large files efficiently.
        The iter_lines() method processes the file line by line
        rather than loading it entirely into memory.
    """
    # Parse S3 URI into bucket and key components
    bucket, key = utils.split_s3_uri(s3_uri)

    # Fetch object from S3
    response = s3_client.get_object(Bucket=bucket, Key=key)

    # Stream the response body line by line for memory efficiency
    body = response["Body"]
    lines = body.iter_lines()  # Generator for lazy evaluation

    # Parse JSONL with S3 URI as context for error messages
    return parse_jsonl(lines, context=s3_uri)


def lambda_handler(event: TaskItem | dict, context):
    """Process Bedrock batch inference outputs and join with original inputs.

    This Lambda function is the final step in the batch inference pipeline.
    It retrieves model outputs from S3, processes them according to the model
    type, joins them with original inputs, and saves the complete dataset.

    The function handles both successful and failed jobs gracefully, ensuring
    that failures in one job don't prevent processing of others.

    Args:
        event: Task configuration from Step Functions containing:
            - model_id: Bedrock model identifier
            - job_arn: ARN of completed batch job
            - input_parquet_path: Location of original input data
            - s3_uri_output: Directory containing job outputs
            - error_message: Optional error if job failed
        context: Lambda runtime context (not used but required by Lambda).

    Returns:
        Dictionary with:
            - output_path: S3 URI of processed Parquet file, or None if job failed.

    Note:
        Failed jobs (those with error_message) are skipped to prevent
        cascading failures in the pipeline. This ensures partial success
        scenarios where some batch jobs succeed while others fail.
    """
    # Validate and normalize input event structure
    task = event if isinstance(event, TaskItem) else TaskItem.model_validate(event)

    # Log job details for debugging and monitoring
    logger.info(f"Postprocessing job:\n{task.model_dump_json(indent=2)}")

    # Only process successful jobs - skip if there was an error
    if not task.error_message:
        # Get the appropriate processor for parsing model-specific output format
        processor = get_processor_for_model_id(task.model_id)

        # Load original input data that was saved during preprocessing
        # This contains the source data with record IDs for joining
        input_df = wr.s3.read_parquet(task.input_parquet_path)

        # Construct path to Bedrock output using job ARN
        # Bedrock saves outputs in a subdirectory named after the job ID
        output_prefix = os.path.join(task.s3_uri_output, task.job_arn.split("/")[-1])
        logger.info(f"Retrieving model output from {output_prefix}")

        # Find the output JSONL file - Bedrock appends '.out' to the input filename
        # There should be exactly one .jsonl.out file per job
        model_output_uri = next(
            iter(
                wr.s3.list_objects(
                    path=output_prefix,
                    suffix=".jsonl.out",  # Bedrock's output file extension
                )
            )
        )
        logger.info(f"Output URI: {model_output_uri}")
        # Parse JSONL output from Bedrock
        output_records = read_jsonl_from_s3(model_output_uri)

        # Process each output record using model-specific processor
        # This extracts the relevant fields (text response or embedding vector)
        processed_outputs = [processor.process_output(r) for r in output_records]

        # Join model outputs with original inputs using record_id
        # This creates a complete dataset with both inputs and outputs
        output_df = pd.DataFrame(processed_outputs).merge(input_df, on="record_id")

        # Construct output path maintaining the same directory structure
        # This makes it easy to correlate inputs and outputs
        output_parquet_path = os.path.join(
            f"s3://{BUCKET_NAME}/batch_output_parquet/",
            *task.input_parquet_path.split("/")[
                -2:
            ],  # Preserve job prefix and chunk number
        )
        logger.info(f"Saving output parquet to {output_parquet_path}")

        # Save the enriched dataset as Parquet for efficient querying
        # Snappy compression provides good balance of size and speed
        wr.s3.to_parquet(
            output_df,
            output_parquet_path,
            index=False,  # Don't save DataFrame index
            compression="snappy",  # Fast compression for analytics workloads
        )
    else:
        # Skip processing for failed jobs to prevent pipeline failures
        # This allows partial success where some chunks succeed
        logger.warning(
            f"Skipping postprocessing due to job error: {task.error_message}"
        )
        output_parquet_path = None

    # Return the output location for downstream processing or None if failed
    # Step Functions can use this to track successful completions
    return {
        "output_path": output_parquet_path,
    }
