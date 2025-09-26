from custom_types import TaskItem
from processor import get_processor_for_model_id
import utils
from typing import Iterable, List, Dict, Union
import awswrangler as wr
import pandas as pd
import boto3
import json
import os


logger = utils.get_logger()
BUCKET_NAME = os.getenv('BUCKET_NAME')

s3_client = boto3.client('s3')


def _normalize_lines(lines: Iterable[Union[str, bytes]]) -> Iterable[str]:
    """Yield UTF-8 decoded lines stripped of trailing newline characters."""

    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        yield line.rstrip('\n')


def parse_jsonl(lines: Iterable[Union[str, bytes]], *, context: str | None = None) -> List[Dict]:
    """Parse an iterable of JSON Lines strings into Python dictionaries.

    Args:
        lines: Iterable yielding each JSON document as a string or bytes value.
        context: Optional label (for example an S3 URI) included in raised errors.

    Returns:
        List of parsed JSON objects in the order they were provided.

    Raises:
        ValueError: If any non-empty line fails JSON decoding.
    """

    records: List[Dict] = []
    for line_number, raw_line in enumerate(_normalize_lines(lines), start=1):
        line = raw_line.strip()
        if not line:
            # Allow blank or whitespace-only lines without failing the whole job.
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            location = f" {context}" if context else ''
            raise ValueError(
                f'Failed to parse JSONL{location} on line {line_number}: {line!r}'
            ) from exc

    return records


def read_jsonl_from_s3(s3_uri: str) -> List[Dict]:
    """Fetch a JSONL object from S3 and parse it into Python dictionaries."""

    bucket, key = utils.split_s3_uri(s3_uri)
    response = s3_client.get_object(Bucket=bucket, Key=key)

    body = response['Body']
    # Use iter_lines to stream the object and avoid loading very large files into memory at once.
    lines = body.iter_lines()

    return parse_jsonl(lines, context=s3_uri)


def lambda_handler(event: TaskItem, context):
    """
    Bedrock batch inference jobs are returned as JSONL files. This postprocessing step is necessary for parsing
    the output files AND joining the result back to the original input record via a join with the record_id.

    Final outputs are saved as Parquet files at the returned S3 paths.
    """

    logger.info(f'Postprocessing job:\n{event}')

    if not event['error_message']:
        processor = get_processor_for_model_id(event['model_id'])
        input_df = wr.s3.read_parquet(event['input_parquet_path'])

        output_prefix = os.path.join(event['s3_uri_output'], event['job_arn'].split('/')[-1])
        logger.info(f'Retrieving model output from {output_prefix}')
        model_output_uri = next(iter(wr.s3.list_objects(
            path=output_prefix,
            suffix='.jsonl.out',
        )))
        logger.info(f'Output URI: {model_output_uri}')
        output_records = read_jsonl_from_s3(model_output_uri)
        processed_outputs = [processor.process_output(r) for r in output_records]

        output_df = pd.DataFrame(processed_outputs).merge(input_df, on='record_id')
        output_parquet_path = os.path.join(f's3://{BUCKET_NAME}/batch_output_parquet/', *event['input_parquet_path'].split('/')[-2:])
        logger.info(f'Saving output parquet to {output_parquet_path}')

        wr.s3.to_parquet(
            output_df,
            output_parquet_path,
            index=False,
            compression='snappy',
        )
    else:
        # if an error occurred, skip processing
        output_parquet_path = None

    return {
        'output_path': output_parquet_path,
    }
