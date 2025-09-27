import json
import os
from collections.abc import Iterable

import awswrangler as wr
import boto3
import pandas as pd
import utils
from custom_types import TaskItem
from processor import get_processor_for_model_id

logger = utils.get_logger()
BUCKET_NAME = os.getenv("BUCKET_NAME")

s3_client = boto3.client("s3")


def _normalize_lines(lines: Iterable[str | bytes]) -> Iterable[str]:
    """Yield UTF-8 decoded lines stripped of trailing newline characters."""

    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        yield line.rstrip("\n")


def parse_jsonl(
    lines: Iterable[str | bytes], *, context: str | None = None
) -> list[dict]:
    """Parse an iterable of JSON Lines strings into Python dictionaries.

    Args:
        lines: Iterable yielding each JSON document as a string or bytes value.
        context: Optional label (for example an S3 URI) included in raised errors.

    Returns:
        List of parsed JSON objects in the order they were provided.

    Raises:
        ValueError: If any non-empty line fails JSON decoding.
    """

    records: list[dict] = []
    for line_number, raw_line in enumerate(_normalize_lines(lines), start=1):
        line = raw_line.strip()
        if not line:
            # Allow blank or whitespace-only lines without failing the whole job.
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            location = f" {context}" if context else ""
            raise ValueError(
                f"Failed to parse JSONL{location} on line {line_number}: {line!r}"
            ) from exc

    return records


def read_jsonl_from_s3(s3_uri: str) -> list[dict]:
    """Fetch a JSONL object from S3 and parse it into Python dictionaries."""

    bucket, key = utils.split_s3_uri(s3_uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)

    body = response["Body"]
    # Use iter_lines to stream the object and avoid loading very large files into memory at once.
    lines = body.iter_lines()

    return parse_jsonl(lines, context=s3_uri)


def lambda_handler(event: TaskItem | dict, context):
    """
    Bedrock batch inference jobs are returned as JSONL files. This postprocessing step is necessary for parsing
    the output files AND joining the result back to the original input record via a join with the record_id.

    Final outputs are saved as Parquet files at the returned S3 paths.
    """

    task = event if isinstance(event, TaskItem) else TaskItem.model_validate(event)

    logger.info(f"Postprocessing job:\n{task.model_dump_json(indent=2)}")

    if not task.error_message:
        processor = get_processor_for_model_id(task.model_id)
        input_df = wr.s3.read_parquet(task.input_parquet_path)

        output_prefix = os.path.join(task.s3_uri_output, task.job_arn.split("/")[-1])
        logger.info(f"Retrieving model output from {output_prefix}")
        model_output_uri = next(
            iter(
                wr.s3.list_objects(
                    path=output_prefix,
                    suffix=".jsonl.out",
                )
            )
        )
        logger.info(f"Output URI: {model_output_uri}")
        output_records = read_jsonl_from_s3(model_output_uri)
        processed_outputs = [processor.process_output(r) for r in output_records]

        output_df = pd.DataFrame(processed_outputs).merge(input_df, on="record_id")
        output_parquet_path = os.path.join(
            f"s3://{BUCKET_NAME}/batch_output_parquet/",
            *task.input_parquet_path.split("/")[-2:],
        )
        logger.info(f"Saving output parquet to {output_parquet_path}")

        wr.s3.to_parquet(
            output_df,
            output_parquet_path,
            index=False,
            compression="snappy",
        )
    else:
        # if an error occurred, skip processing
        output_parquet_path = None

    return {
        "output_path": output_parquet_path,
    }
