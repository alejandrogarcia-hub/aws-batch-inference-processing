"""Unit tests for lambda/postprocess.py.

This module provides comprehensive test coverage for all functions in postprocess.py,
following AAA (Arrange-Act-Assert) pattern and unit testing best practices.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Provide lightweight stubs for AWS dependencies before importing the Lambda module
if "awswrangler" not in sys.modules:
    awswrangler_stub = types.ModuleType("awswrangler")
    awswrangler_stub.s3 = types.SimpleNamespace()
    sys.modules["awswrangler"] = awswrangler_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")

    class _StubDataFrame:
        def __init__(self, *args, **kwargs):
            self.merge = MagicMock()

    pandas_stub.DataFrame = _StubDataFrame
    sys.modules["pandas"] = pandas_stub

if "boto3" not in sys.modules:
    boto3_stub = types.ModuleType("boto3")
    boto3_stub.client = Mock()
    boto3_stub.resource = Mock()
    sys.modules["boto3"] = boto3_stub

# Ensure the Lambda source directory is importable
LAMBDA_SRC = Path(__file__).resolve().parents[1]
if str(LAMBDA_SRC) not in sys.path:
    sys.path.append(str(LAMBDA_SRC))

# Import after stubs are in place
import postprocess  # noqa: E402
from postprocess import (  # noqa: E402
    _normalize_lines,
    lambda_handler,
    parse_jsonl,
    read_jsonl_from_s3,
)
from custom_types import TaskItem  # noqa: E402


class TestNormalizeLines(unittest.TestCase):
    """Test cases for _normalize_lines following AAA pattern."""

    def test_should_strip_trailing_newlines_from_string_input(self):
        """It should strip only trailing newline characters (\n) from string inputs."""
        # Arrange
        input_lines = ["line1\n", "line2\r\n", "line3\n\n", "line4"]
        expected = ["line1", "line2\r", "line3", "line4"]  # rstrip('\n') only removes trailing \n

        # Act
        result = list(_normalize_lines(input_lines))

        # Assert
        self.assertEqual(result, expected)

    def test_should_only_strip_newline_not_carriage_return(self):
        """It should only strip \\n, preserving \\r characters."""
        # Arrange
        input_lines = ["text\r", "text\r\n", "text\n", "text\r\r\n"]
        expected = ["text\r", "text\r", "text", "text\r\r"]

        # Act
        result = list(_normalize_lines(input_lines))

        # Assert
        self.assertEqual(result, expected)

    def test_should_decode_bytes_to_utf8(self):
        """It should decode bytes input to UTF-8 strings."""
        # Arrange
        input_lines = [b"line1\n", b"line2", b"\xc3\xa9\n"]  # é in UTF-8
        expected = ["line1", "line2", "é"]

        # Act
        result = list(_normalize_lines(input_lines))

        # Assert
        self.assertEqual(result, expected)

    def test_should_handle_mixed_string_and_bytes_input(self):
        """It should process both string and bytes in the same iterable."""
        # Arrange
        input_lines = ["string1\n", b"bytes1\n", "string2", b"bytes2"]
        expected = ["string1", "bytes1", "string2", "bytes2"]

        # Act
        result = list(_normalize_lines(input_lines))

        # Assert
        self.assertEqual(result, expected)

    def test_should_preserve_empty_lines_but_strip_newlines(self):
        """It should preserve empty lines while stripping newlines."""
        # Arrange
        input_lines = ["", "\n", b"", b"\n"]
        expected = ["", "", "", ""]

        # Act
        result = list(_normalize_lines(input_lines))

        # Assert
        self.assertEqual(result, expected)

    def test_should_process_generator_input_lazily(self):
        """It should handle generator input for memory efficiency."""
        # Arrange
        def line_generator():
            yield "line1\n"
            yield b"line2\n"
        expected = ["line1", "line2"]

        # Act
        result = list(_normalize_lines(line_generator()))

        # Assert
        self.assertEqual(result, expected)


class TestParseJsonl(unittest.TestCase):
    """Test cases for parse_jsonl following AAA pattern and best practices."""

    def test_should_return_empty_list_for_empty_input(self):
        """It should return an empty list when given empty input."""
        # Arrange
        input_lines = []
        expected = []

        # Act
        result = parse_jsonl(input_lines)

        # Assert
        self.assertEqual(result, expected)

    def test_should_parse_valid_json_lines(self):
        """It should parse multiple valid JSON lines into dictionaries."""
        # Arrange
        input_lines = ['{"id": 1, "name": "test"}', '{"id": 2, "value": 42}']
        expected = [{"id": 1, "name": "test"}, {"id": 2, "value": 42}]

        # Act
        result = parse_jsonl(input_lines)

        # Assert
        self.assertEqual(result, expected)

    def test_should_skip_blank_lines(self):
        """It should skip blank and whitespace-only lines."""
        # Arrange
        input_lines = ['{"id": 1}', "   ", "", "\n", '{"id": 2}']
        expected = [{"id": 1}, {"id": 2}]

        # Act
        result = parse_jsonl(input_lines)

        # Assert
        self.assertEqual(result, expected)

    def test_should_handle_unicode_characters(self):
        """It should correctly parse JSON containing Unicode characters."""
        # Arrange
        input_lines = ['{"text": "Hello 世界"}', '{"emoji": "🎉"}']
        expected = [{"text": "Hello 世界"}, {"emoji": "🎉"}]

        # Act
        result = parse_jsonl(input_lines)

        # Assert
        self.assertEqual(result, expected)

    def test_should_parse_nested_json_structures(self):
        """It should parse complex nested JSON objects and arrays."""
        # Arrange
        input_lines = [
            '{"outer": {"inner": {"deep": "value"}}}',
            '{"array": [1, 2, {"key": "value"}]}'
        ]

        # Act
        result = parse_jsonl(input_lines)

        # Assert
        self.assertEqual(result[0]["outer"]["inner"]["deep"], "value")
        self.assertEqual(result[1]["array"][2]["key"], "value")

    def test_should_raise_valueerror_for_invalid_json(self):
        """It should raise ValueError with line number for invalid JSON."""
        # Arrange
        input_lines = ['{"valid": 1}', 'not-json', '{"valid": 2}']

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            parse_jsonl(input_lines)

        error_message = str(context.exception)
        self.assertIn("line 2", error_message)
        self.assertIn("not-json", error_message)

    def test_should_include_context_in_error_message(self):
        """It should include provided context in error messages."""
        # Arrange
        input_lines = ["invalid-json"]
        context_info = "s3://bucket/file.jsonl"

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            parse_jsonl(input_lines, context=context_info)

        error_message = str(context.exception)
        self.assertIn(context_info, error_message)
        self.assertIn("line 1", error_message)

    def test_should_handle_escaped_special_characters(self):
        """It should correctly parse escaped special characters in JSON."""
        # Arrange
        input_lines = [
            '{"text": "Line\\nbreak"}',
            '{"quote": "\\"quoted\\""}',
            '{"tab": "Tab\\there"}'
        ]

        # Act
        result = parse_jsonl(input_lines)

        # Assert
        self.assertEqual(result[0]["text"], "Line\nbreak")
        self.assertEqual(result[1]["quote"], '"quoted"')
        self.assertEqual(result[2]["tab"], "Tab\there")

    # Backward compatibility tests from original file
    def test_parses_multiple_valid_lines(self) -> None:
        lines = ['{"recordId": "1"}', '{"recordId": "2", "value": 42}']

        result = parse_jsonl(lines, context="s3://bucket/key")

        self.assertEqual(
            result,
            [
                {"recordId": "1"},
                {"recordId": "2", "value": 42},
            ],
        )

    def test_ignores_blank_lines(self) -> None:
        lines = [' {"recordId": "1"} ', "   ", "\n", '{"recordId": "2"}']

        result = parse_jsonl(lines)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["recordId"], "1")
        self.assertEqual(result[1]["recordId"], "2")

    def test_accepts_bytes_input(self) -> None:
        lines = [b'{"recordId": "bytes"}', b' {"value": true} ']

        result = parse_jsonl(lines)

        self.assertEqual(result[0]["recordId"], "bytes")
        self.assertTrue(result[1]["value"])

    def test_raises_value_error_with_context(self) -> None:
        with self.assertRaises(ValueError) as err:
            parse_jsonl(["not-json"], context="s3://bucket/bad-file")

        self.assertIn("s3://bucket/bad-file", str(err.exception))
        self.assertIn("line 1", str(err.exception))


class TestReadJsonlFromS3(unittest.TestCase):
    """Test cases for read_jsonl_from_s3 following AAA pattern."""

    @patch('postprocess.s3_client')
    @patch('postprocess.utils.split_s3_uri')
    def test_should_read_and_parse_jsonl_from_s3(self, mock_split_uri, mock_s3_client):
        """It should successfully read and parse JSONL content from S3."""
        # Arrange
        s3_uri = "s3://test-bucket/data/file.jsonl"
        mock_split_uri.return_value = ("test-bucket", "data/file.jsonl")

        mock_body = MagicMock()
        mock_body.iter_lines.return_value = [
            b'{"recordId": "123", "status": "success"}',
            b'{"recordId": "456", "status": "pending"}'
        ]
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        expected = [
            {"recordId": "123", "status": "success"},
            {"recordId": "456", "status": "pending"}
        ]

        # Act
        result = read_jsonl_from_s3(s3_uri)

        # Assert
        self.assertEqual(result, expected)
        mock_split_uri.assert_called_once_with(s3_uri)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="data/file.jsonl"
        )

    @patch('postprocess.s3_client')
    @patch('postprocess.utils.split_s3_uri')
    def test_should_handle_empty_s3_file(self, mock_split_uri, mock_s3_client):
        """It should return empty list for empty S3 file."""
        # Arrange
        s3_uri = "s3://bucket/empty.jsonl"
        mock_split_uri.return_value = ("bucket", "empty.jsonl")

        mock_body = MagicMock()
        mock_body.iter_lines.return_value = []
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        expected = []

        # Act
        result = read_jsonl_from_s3(s3_uri)

        # Assert
        self.assertEqual(result, expected)

    @patch('postprocess.s3_client')
    @patch('postprocess.utils.split_s3_uri')
    def test_should_propagate_s3_client_errors(self, mock_split_uri, mock_s3_client):
        """It should propagate S3 client errors (NoSuchKey, AccessDenied, etc.)."""
        # Arrange
        s3_uri = "s3://bucket/missing.jsonl"
        mock_split_uri.return_value = ("bucket", "missing.jsonl")

        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, "GetObject")

        # Act & Assert
        with self.assertRaises(ClientError) as context:
            read_jsonl_from_s3(s3_uri)

        self.assertEqual(context.exception.response["Error"]["Code"], "NoSuchKey")

    @patch('postprocess.s3_client')
    @patch('postprocess.utils.split_s3_uri')
    def test_should_include_s3_uri_in_parse_errors(self, mock_split_uri, mock_s3_client):
        """It should include S3 URI context in JSON parse errors."""
        # Arrange
        s3_uri = "s3://bucket/malformed.jsonl"
        mock_split_uri.return_value = ("bucket", "malformed.jsonl")

        mock_body = MagicMock()
        mock_body.iter_lines.return_value = [b'{"valid": true}', b'invalid-json']
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            read_jsonl_from_s3(s3_uri)

        error_message = str(context.exception)
        self.assertIn(s3_uri, error_message)
        self.assertIn("line 2", error_message)


class TestLambdaHandler(unittest.TestCase):
    """Test cases for lambda_handler following AAA pattern."""

    def setUp(self):
        """Set up common test fixtures."""
        os.environ["BUCKET_NAME"] = "test-bucket"
        self.mock_context = MagicMock()

    def tearDown(self):
        """Clean up after tests."""
        if "BUCKET_NAME" in os.environ:
            del os.environ["BUCKET_NAME"]

    @patch('postprocess.logger')
    @patch('postprocess.get_processor_for_model_id')
    @patch('postprocess.wr.s3')
    @patch('postprocess.read_jsonl_from_s3')
    def test_should_process_successful_job(
        self, mock_read_jsonl, mock_wr_s3, mock_get_processor, _mock_logger
    ):
        """It should process a successful batch job and save results."""
        # Arrange
        task_input = {
            "job_arn": "arn:aws:bedrock:us-east-1:123456789012:job/job-123",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "input_parquet_path": "s3://test-bucket/input/001.parquet",
            "s3_uri_output": "s3://test-bucket/output/",
            "status": "Completed",
            "error_message": None,
            "task_token": "token-123"
        }

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process_output.return_value = {
            "record_id": "rec-1",
            "response": "Generated text"
        }
        mock_get_processor.return_value = mock_processor

        # Mock S3 operations
        mock_wr_s3.list_objects.return_value = [
            "s3://test-bucket/output/job-123/output.jsonl.out"
        ]

        mock_input_df = MagicMock()
        mock_wr_s3.read_parquet.return_value = mock_input_df

        mock_read_jsonl.return_value = [
            {"recordId": "rec-1", "modelOutput": {"content": [{"text": "Generated text"}]}}
        ]

        # Mock DataFrame operations - patch pd.DataFrame in the test
        if not hasattr(postprocess.pd, "DataFrame"):
            postprocess.pd.DataFrame = MagicMock()  # type: ignore[attr-defined]

        with patch('postprocess.pd.DataFrame') as mock_df_class:
            mock_df_instance = mock_df_class.return_value
            mock_merged_df = MagicMock()
            mock_df_instance.merge.return_value = mock_merged_df

            # Act
            result = lambda_handler(task_input, self.mock_context)

            # Assert
            self.assertIn("output_path", result)
            self.assertIsNotNone(result["output_path"])
            self.assertIn("batch_output_parquet", result["output_path"])

            mock_get_processor.assert_called_once_with("anthropic.claude-3-haiku-20240307-v1:0")
            mock_wr_s3.read_parquet.assert_called_once_with("s3://test-bucket/input/001.parquet")
            mock_wr_s3.to_parquet.assert_called_once()
            mock_df_instance.merge.assert_called_once_with(mock_input_df, on="record_id")

    @patch('postprocess.logger')
    def test_should_skip_processing_for_failed_job(self, mock_logger):
        """It should skip processing when job has an error message."""
        # Arrange
        task_input = {
            "job_arn": "arn:aws:bedrock:us-east-1:123456789012:job/job-failed",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "input_parquet_path": "s3://test-bucket/input/002.parquet",
            "s3_uri_output": "s3://test-bucket/output/",
            "status": "Failed",
            "error_message": "Job failed due to invalid input format",
            "task_token": "token-failed"
        }

        # Act
        result = lambda_handler(task_input, self.mock_context)

        # Assert
        self.assertIsNone(result["output_path"])
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Job failed due to invalid input format", warning_call)

    def test_should_validate_dict_input_to_task_item(self):
        """It should validate dictionary input and convert to TaskItem."""
        # Arrange
        task_dict = {
            "job_arn": "arn:aws:bedrock:us-east-1:123456789012:job/test",
            "model_id": "amazon.titan-embed-text-v2:0",
            "input_parquet_path": "s3://bucket/input.parquet",
            "s3_uri_output": "s3://bucket/output/",
            "status": "InProgress",
            "task_token": "test-token"
        }

        # Act
        with patch('postprocess.logger'):
            # Test validation by creating TaskItem
            task_item = TaskItem.model_validate(task_dict)

        # Assert
        self.assertEqual(task_item.job_arn, task_dict["job_arn"])
        self.assertEqual(task_item.model_id, task_dict["model_id"])
        self.assertIsNone(task_item.error_message)

    @patch('postprocess.logger')
    @patch('postprocess.get_processor_for_model_id')
    @patch('postprocess.wr.s3')
    def test_should_raise_error_when_output_file_not_found(
        self, mock_wr_s3, mock_get_processor, _mock_logger
    ):
        """It should raise StopIteration when output file is not found."""
        # Arrange
        task_input = {
            "job_arn": "arn:aws:bedrock:us-east-1:123456789012:job/no-output",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "input_parquet_path": "s3://test-bucket/input/003.parquet",
            "s3_uri_output": "s3://test-bucket/output/",
            "status": "Completed",
            "error_message": None,
            "task_token": "token-no-output"
        }

        mock_get_processor.return_value = MagicMock()
        mock_wr_s3.read_parquet.return_value = MagicMock()
        mock_wr_s3.list_objects.return_value = iter([])  # Empty iterator

        # Act & Assert
        with self.assertRaises(StopIteration):
            lambda_handler(task_input, self.mock_context)

    def test_should_handle_task_item_instance_input(self):
        """It should handle TaskItem instance directly without validation."""
        # Arrange
        task_item = TaskItem(
            job_arn="arn:aws:bedrock:us-east-1:123456789012:job/direct",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            input_parquet_path="s3://test-bucket/input/direct.parquet",
            s3_uri_output="s3://test-bucket/output/",
            status="Failed",
            error_message="Direct task error",
            task_token="token-direct"
        )

        # Act
        with patch('postprocess.logger') as mock_logger:
            result = lambda_handler(task_item, self.mock_context)

        # Assert
        self.assertIsNone(result["output_path"])
        mock_logger.info.assert_called()


if __name__ == "__main__":
    unittest.main()