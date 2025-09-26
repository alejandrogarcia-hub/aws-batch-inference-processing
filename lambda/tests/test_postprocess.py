"""Unit tests for helpers in lambda/postprocess.py."""

from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest


# Provide lightweight stubs for AWS dependencies before importing the Lambda module.
if 'awswrangler' not in sys.modules:
    awswrangler_stub = types.ModuleType('awswrangler')
    awswrangler_stub.s3 = types.SimpleNamespace()  # pragma: no cover - placeholder for runtime APIs
    sys.modules['awswrangler'] = awswrangler_stub


if 'pandas' not in sys.modules:
    sys.modules['pandas'] = types.ModuleType('pandas')


class _StubS3Client:
    def get_object(self, *args, **kwargs):  # pragma: no cover - not used in these unit tests
        raise NotImplementedError("S3 client stub does not implement get_object")


def _boto3_client_stub(service_name, *args, **kwargs):  # pragma: no cover - simple guard
    if service_name == 's3':
        return _StubS3Client()
    raise NotImplementedError(f"Unsupported service stub: {service_name}")


if 'boto3' not in sys.modules:
    boto3_stub = types.ModuleType('boto3')
    boto3_stub.client = _boto3_client_stub
    boto3_stub.resource = lambda *args, **kwargs: None  # pragma: no cover
    sys.modules['boto3'] = boto3_stub


# Ensure the Lambda source directory is importable when running tests locally.
LAMBDA_SRC = Path(__file__).resolve().parents[1]
if str(LAMBDA_SRC) not in sys.path:
    sys.path.append(str(LAMBDA_SRC))


from postprocess import parse_jsonl  # noqa: E402  pylint: disable=wrong-import-position


class ParseJsonlTests(unittest.TestCase):
    """Validate robustness of the JSONL parsing helper."""

    def test_parses_multiple_valid_lines(self) -> None:
        lines = ['{"recordId": "1"}', '{"recordId": "2", "value": 42}']

        result = parse_jsonl(lines, context='s3://bucket/key')

        self.assertEqual(
            result,
            [
                {"recordId": "1"},
                {"recordId": "2", "value": 42},
            ],
        )

    def test_ignores_blank_lines(self) -> None:
        lines = [' {"recordId": "1"} ', '   ', '\n', '{"recordId": "2"}']

        result = parse_jsonl(lines)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['recordId'], '1')
        self.assertEqual(result[1]['recordId'], '2')

    def test_accepts_bytes_input(self) -> None:
        lines = [b'{"recordId": "bytes"}', b' {"value": true} ']

        result = parse_jsonl(lines)

        self.assertEqual(result[0]['recordId'], 'bytes')
        self.assertTrue(result[1]['value'])

    def test_raises_value_error_with_context(self) -> None:
        with self.assertRaises(ValueError) as err:
            parse_jsonl(['not-json'], context='s3://bucket/bad-file')

        self.assertIn('s3://bucket/bad-file', str(err.exception))
        self.assertIn('line 1', str(err.exception))


if __name__ == '__main__':
    unittest.main()
