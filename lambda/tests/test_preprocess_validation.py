"""Unit tests covering validation helpers in lambda/preprocess.py."""

from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest
from unittest import mock


import os


# Stub heavy dependencies before importing the Lambda module.
if 'awswrangler' not in sys.modules:
    awswrangler_stub = types.ModuleType('awswrangler')
    awswrangler_stub.s3 = types.SimpleNamespace()
    sys.modules['awswrangler'] = awswrangler_stub

if 'pandas' not in sys.modules:
    sys.modules['pandas'] = types.ModuleType('pandas')

if 'datasets' not in sys.modules:
    datasets_stub = types.ModuleType('datasets')
    datasets_stub.load_dataset = lambda *args, **kwargs: None  # pragma: no cover - not exercised in these tests
    sys.modules['datasets'] = datasets_stub

awswrangler_module = sys.modules.get('awswrangler')
if awswrangler_module is None:
    awswrangler_module = types.ModuleType('awswrangler')
    sys.modules['awswrangler'] = awswrangler_module

if not hasattr(awswrangler_module, 's3'):
    awswrangler_module.s3 = types.SimpleNamespace()

if not hasattr(awswrangler_module.s3, 'read_csv'):
    awswrangler_module.s3.read_csv = lambda *args, **kwargs: []

if not hasattr(awswrangler_module.s3, 'read_parquet'):
    awswrangler_module.s3.read_parquet = lambda *args, **kwargs: []

if not hasattr(awswrangler_module.s3, 'to_parquet'):
    awswrangler_module.s3.to_parquet = lambda *args, **kwargs: None


if 'boto3' not in sys.modules:
    boto3_stub = types.ModuleType('boto3')
    boto3_stub.client = lambda *args, **kwargs: None  # pragma: no cover
    boto3_stub.resource = lambda *args, **kwargs: None  # pragma: no cover
    sys.modules['boto3'] = boto3_stub


LAMBDA_SRC = Path(__file__).resolve().parents[1]
if str(LAMBDA_SRC) not in sys.path:
    sys.path.append(str(LAMBDA_SRC))


from preprocess import resolve_prompt_configuration, validate_chunk_columns, lambda_handler  # noqa: E402  pylint: disable=wrong-import-position


os.environ.setdefault('BUCKET_NAME', 'test-bucket')


class ResolvePromptConfigurationTests(unittest.TestCase):
    """validate the prompt configuration helper."""

    def test_embedding_model_returns_empty_configuration(self) -> None:
        template, fields = resolve_prompt_configuration('embedding', {})
        self.assertIsNone(template)
        self.assertEqual(fields, set())

    def test_missing_prompt_id_raises(self) -> None:
        with self.assertRaises(ValueError) as err:
            resolve_prompt_configuration('text', {'prompt_id': None})

        self.assertIn('prompt_id', str(err.exception))

    def test_unknown_prompt_id_lists_available(self) -> None:
        with self.assertRaises(ValueError) as err:
            resolve_prompt_configuration('text', {'prompt_id': 'does_not_exist'})

        self.assertIn('does_not_exist', str(err.exception))

    def test_known_prompt_returns_required_fields(self) -> None:
        template, fields = resolve_prompt_configuration('text', {'prompt_id': 'joke_about_topic'})
        self.assertIn('{topic', template)
        self.assertEqual(fields, {'topic'})


class ValidateChunkColumnsTests(unittest.TestCase):
    """Check dataset column validation for embeddings and text prompts."""

    def test_embedding_requires_input_text(self) -> None:
        with self.assertRaises(ValueError) as err:
            validate_chunk_columns(
                'embedding',
                {'not_input_text'},
                required_prompt_fields=set(),
                job_name_prefix='demo',
                prompt_id=None,
            )

        self.assertIn('input_text', str(err.exception))

    def test_embedding_passes_with_input_text(self) -> None:
        validate_chunk_columns(
            'embedding',
            {'input_text', 'record_id'},
            required_prompt_fields=set(),
            job_name_prefix='demo',
            prompt_id=None,
        )  # should not raise

    def test_text_missing_required_fields_raises(self) -> None:
        with self.assertRaises(ValueError) as err:
            validate_chunk_columns(
                'text',
                {'record_id'},
                required_prompt_fields={'topic'},
                job_name_prefix='demo',
                prompt_id='joke_about_topic',
            )

        self.assertIn('topic', str(err.exception))

    def test_text_with_required_fields_passes(self) -> None:
        validate_chunk_columns(
            'text',
            {'record_id', 'topic'},
            required_prompt_fields={'topic'},
            job_name_prefix='demo',
            prompt_id='joke_about_topic',
        )  # should not raise


class _StubDataFrame:
    def __init__(self, records):
        self._records = [dict(r) for r in records]
        self.columns = list(self._records[0].keys()) if self._records else []

    def __len__(self):
        return len(self._records)

    def to_dict(self, orient):
        assert orient == 'records'
        return [dict(r) for r in self._records]

    def head(self, n):
        return _StubDataFrame(self._records[:n])

    def drop(self, columns):
        return _StubDataFrame([
            {k: v for k, v in row.items() if k not in columns}
            for row in self._records
        ])

    def __setitem__(self, key, values):
        if isinstance(values, list) and len(values) != len(self._records):
            raise ValueError('Length mismatch when assigning column values in stub dataframe.')
        if isinstance(values, list):
            iterable = values
        else:
            iterable = [values] * len(self._records)

        for row, value in zip(self._records, iterable):
            row[key] = value

        if key not in self.columns:
            self.columns.append(key)


class LambdaHandlerMaxRecordsTests(unittest.TestCase):
    """Ensure the preprocess handler respects max_records_total overrides."""

    def setUp(self) -> None:
        self.load_patch = mock.patch('preprocess.utils.load_files_in_chunks')
        self.write_patch = mock.patch('preprocess.write_jsonl_to_s3', return_value='s3://bucket/key')
        self.parquet_patch = mock.patch('preprocess.wr.s3.to_parquet')
        self.uuid_patch = mock.patch('preprocess.uuid4', side_effect=(f'id-{i}' for i in range(100000)))

    def tearDown(self) -> None:
        mock.patch.stopall()

    def test_without_limit_processes_all_chunks(self) -> None:
        load_mock = self.load_patch.start()
        self.write_patch.start()
        self.parquet_patch.start()
        self.uuid_patch.start()

        def _yield_chunks(*args, **kwargs):
            for chunk_index in range(60):  # 60 chunks of 1000 records => 60k records raw
                yield chunk_index, _StubDataFrame({'input_text': f'text-{chunk_index}-{i}'} for i in range(1000))

        load_mock.side_effect = _yield_chunks

        event = {
            's3_uri': 's3://source/input.csv',
            'job_name_prefix': 'test-batch',
            'model_id': 'amazon.titan-embed-text-v2:0',
        }

        result = lambda_handler(event, object())

        self.assertEqual(len(result['jobs']), 60)
        load_mock.assert_called_once()

    def test_respects_max_records_total(self) -> None:
        load_mock = self.load_patch.start()
        self.write_patch.start()
        self.parquet_patch.start()
        self.uuid_patch.start()

        def _yield_chunks(*args, **kwargs):
            for chunk_index in range(10):
                yield chunk_index, _StubDataFrame({'input_text': f'text-{chunk_index}-{i}'} for i in range(1000))

        load_mock.side_effect = _yield_chunks

        event = {
            's3_uri': 's3://source/input.csv',
            'job_name_prefix': 'test-batch',
            'model_id': 'amazon.titan-embed-text-v2:0',
            'max_records_total': 2500,
        }

        result = lambda_handler(event, object())

        self.assertEqual(len(result['jobs']), 3)  # two full chunks + one truncated chunk
        load_mock.assert_called_once()


if __name__ == '__main__':
    unittest.main()
