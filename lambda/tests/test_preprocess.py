"""Comprehensive unit tests for lambda/preprocess.py.

This module provides complete test coverage for all functions in preprocess.py,
following AAA (Arrange-Act-Assert) pattern and unit testing best practices.
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch

# Stub heavy dependencies before importing the Lambda module
if "awswrangler" not in sys.modules:
    awswrangler_stub = types.ModuleType("awswrangler")
    awswrangler_stub.s3 = types.SimpleNamespace(
        to_parquet=Mock(),
        read_parquet=Mock(),
        list_objects=Mock(),
    )
    sys.modules["awswrangler"] = awswrangler_stub


class _StubDataFrame:
    def __init__(self, records=None, *_, **__):
        self.merge = Mock()
        self._records = []
        self._columns = []

        if records is None:
            records = []
        elif isinstance(records, dict):
            keys = list(records.keys())
            length = len(next(iter(records.values()), []))
            self._records = [
                {key: records[key][i] for key in keys}
                for i in range(length)
            ]
            records = []
        elif hasattr(records, '__iter__') and not isinstance(records, (str, bytes)):
            records = list(records)
        else:
            records = [records]

        if records:
            self._records.extend(dict(r) for r in records)
        self._columns = list(self._records[0].keys()) if self._records else []

    @property
    def columns(self):
        return list(self._columns)

    @columns.setter
    def columns(self, value):
        self._columns = list(value)

    def __len__(self):
        return len(self._records)

    def to_dict(self, orient):
        assert orient == "records"
        return [dict(r) for r in self._records]

    def head(self, n):
        return _StubDataFrame(self._records[:n])

    def drop(self, columns):
        return _StubDataFrame(
            [
                {k: v for k, v in row.items() if k not in columns}
                for row in self._records
            ]
        )

    def __setitem__(self, key, values):
        if isinstance(values, list) and len(values) != len(self._records):
            raise ValueError(
                "Length mismatch when assigning column values in stub dataframe."
            )

        iterable = values if isinstance(values, list) else [values] * len(self._records)

        for row, value in zip(self._records, iterable):
            row[key] = value

        if key not in self._columns:
            self._columns.append(key)

pandas_module = sys.modules.get("pandas")
if pandas_module is None:
    pandas_module = types.ModuleType("pandas")
    sys.modules["pandas"] = pandas_module

setattr(pandas_module, "DataFrame", _StubDataFrame)

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

if "boto3" not in sys.modules:
    boto3_stub = types.ModuleType("boto3")
    boto3_stub.client = lambda *args, **kwargs: Mock()
    boto3_stub.resource = lambda *args, **kwargs: None
    sys.modules["boto3"] = boto3_stub

# Ensure the Lambda source directory is importable
LAMBDA_SRC = Path(__file__).resolve().parents[1]
if str(LAMBDA_SRC) not in sys.path:
    sys.path.append(str(LAMBDA_SRC))

# Import pandas after stubbing
import pandas as pd  # noqa: E402

# Import the module and functions for testing
import preprocess  # noqa: E402
from preprocess import (  # noqa: E402
    _parse_optional_int,
    lambda_handler,
    resolve_prompt_configuration,
    validate_chunk_columns,
    write_jsonl_to_s3,
)


class TestParseOptionalInt(unittest.TestCase):
    """Test cases for _parse_optional_int following AAA pattern."""

    def test_should_return_none_for_none_input(self):
        """It should return None when input is None."""
        # Arrange
        input_value = None

        # Act
        result = _parse_optional_int(input_value)

        # Assert
        self.assertIsNone(result)

    def test_should_return_none_for_empty_string(self):
        """It should return None for empty string."""
        # Arrange
        test_cases = ["", "   ", "\t", "\n"]

        for input_value in test_cases:
            # Act
            result = _parse_optional_int(input_value)

            # Assert
            self.assertIsNone(result, f"Failed for input: {repr(input_value)}")

    def test_should_return_none_for_none_string(self):
        """It should return None for string 'none' (case insensitive)."""
        # Arrange
        test_cases = ["none", "None", "NONE", "NoNe"]

        for input_value in test_cases:
            # Act
            result = _parse_optional_int(input_value)

            # Assert
            self.assertIsNone(result, f"Failed for input: {input_value}")

    def test_should_parse_valid_integer_strings(self):
        """It should parse valid integer strings."""
        # Arrange
        test_cases = [
            ("100", 100),
            ("0", 0),
            ("-42", -42),
            ("  123  ", 123),  # With whitespace
            ("999999", 999999),
        ]

        for input_str, expected in test_cases:
            # Act
            result = _parse_optional_int(input_str)

            # Assert
            self.assertEqual(result, expected, f"Failed for input: {input_str}")

    def test_should_parse_integer_values(self):
        """It should handle actual integer inputs."""
        # Arrange
        test_cases = [100, 0, -42, 999999]

        for input_value in test_cases:
            # Act
            result = _parse_optional_int(input_value)

            # Assert
            self.assertEqual(result, input_value)

    def test_should_raise_valueerror_for_invalid_input(self):
        """It should raise ValueError for non-integer strings."""
        # Arrange
        invalid_inputs = ["abc", "12.34", "1e10", "infinity", "NaN"]

        for input_value in invalid_inputs:
            # Act & Assert
            with self.assertRaises(ValueError) as context:
                _parse_optional_int(input_value)

            self.assertIn("Expected an integer value", str(context.exception))


class TestResolvePromptConfiguration(unittest.TestCase):
    """Test cases for resolve_prompt_configuration following AAA pattern."""

    def test_should_return_empty_for_embedding_model(self):
        """It should return empty configuration for embedding models."""
        # Arrange
        model_type = "embedding"
        event = {"prompt_id": "some_prompt"}

        # Act
        template, fields = resolve_prompt_configuration(model_type, event)

        # Assert
        self.assertIsNone(template)
        self.assertEqual(fields, set())

    def test_should_raise_error_for_text_model_without_prompt_id(self):
        """It should raise ValueError when text model lacks prompt_id."""
        # Arrange
        model_type = "text"
        test_cases = [
            {"prompt_id": None},
            {},  # Missing prompt_id
            {"prompt_id": ""},
        ]

        for event in test_cases:
            # Act & Assert
            with self.assertRaises(ValueError) as context:
                resolve_prompt_configuration(model_type, event)

            self.assertIn("prompt_id", str(context.exception))

    @patch('preprocess.pt.prompt_id_to_template', {"prompt1": "Template 1", "prompt2": "Template 2", "prompt3": "Template 3"})
    def test_should_raise_error_for_unknown_prompt_id(self):
        """It should raise ValueError and list available prompts for unknown ID."""
        # Arrange
        model_type = "text"
        event = {"prompt_id": "unknown_prompt"}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            resolve_prompt_configuration(model_type, event)

        error_msg = str(context.exception)
        self.assertIn("unknown_prompt", error_msg)
        self.assertIn("prompt1", error_msg)
        self.assertIn("prompt2", error_msg)
        self.assertIn("prompt3", error_msg)

    @patch('preprocess.pt.prompt_id_to_template')
    def test_should_extract_fields_from_template(self, mock_templates):
        """It should extract placeholder fields from prompt template."""
        # Arrange
        mock_templates.__getitem__.return_value = "Tell me about {topic} in {style} style"
        model_type = "text"
        event = {"prompt_id": "test_prompt"}

        # Act
        template, fields = resolve_prompt_configuration(model_type, event)

        # Assert
        self.assertEqual(template, "Tell me about {topic} in {style} style")
        self.assertEqual(fields, {"topic", "style"})

    @patch('preprocess.pt.prompt_id_to_template')
    def test_should_handle_complex_templates(self, mock_templates):
        """It should handle templates with complex formatting."""
        # Arrange
        mock_templates.__getitem__.return_value = (
            "User: {user_input}\n"
            "Context: {context:.100}\n"
            "Format: {format!s}\n"
            "Plain text without placeholder"
        )
        model_type = "text"
        event = {"prompt_id": "complex"}

        # Act
        template, fields = resolve_prompt_configuration(model_type, event)

        # Assert
        self.assertEqual(fields, {"user_input", "context", "format"})


class TestValidateChunkColumns(unittest.TestCase):
    """Test cases for validate_chunk_columns following AAA pattern."""

    def test_should_require_input_text_for_embedding_model(self):
        """It should require 'input_text' column for embedding models."""
        # Arrange
        model_type = "embedding"
        columns = {"text", "id", "metadata"}  # Missing input_text

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            validate_chunk_columns(
                model_type,
                columns,
                required_prompt_fields=set(),
                job_name_prefix="test-job",
                prompt_id=None,
            )

        self.assertIn("input_text", str(context.exception))
        self.assertIn("test-job", str(context.exception))

    def test_should_pass_with_input_text_for_embedding(self):
        """It should pass validation when input_text is present for embeddings."""
        # Arrange
        model_type = "embedding"
        columns = {"input_text", "record_id", "metadata"}

        # Act & Assert (should not raise)
        validate_chunk_columns(
            model_type,
            columns,
            required_prompt_fields=set(),
            job_name_prefix="test-job",
            prompt_id=None,
        )

    def test_should_require_prompt_fields_for_text_model(self):
        """It should require all prompt template fields for text models."""
        # Arrange
        model_type = "text"
        columns = {"record_id", "context"}  # Missing 'topic'
        required_fields = {"topic", "context"}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            validate_chunk_columns(
                model_type,
                columns,
                required_prompt_fields=required_fields,
                job_name_prefix="test-job",
                prompt_id="test_prompt",
            )

        error_msg = str(context.exception)
        self.assertIn("topic", error_msg)
        self.assertIn("test_prompt", error_msg)

    def test_should_pass_with_all_required_fields_for_text(self):
        """It should pass when all required fields are present for text models."""
        # Arrange
        model_type = "text"
        columns = {"record_id", "topic", "context", "extra_field"}
        required_fields = {"topic", "context"}

        # Act & Assert (should not raise)
        validate_chunk_columns(
            model_type,
            columns,
            required_prompt_fields=required_fields,
            job_name_prefix="test-job",
            prompt_id="test_prompt",
        )

    def test_should_handle_multiple_missing_fields(self):
        """It should report all missing fields in error message."""
        # Arrange
        model_type = "text"
        columns = {"record_id"}
        required_fields = {"field1", "field2", "field3"}

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            validate_chunk_columns(
                model_type,
                columns,
                required_prompt_fields=required_fields,
                job_name_prefix="test-job",
                prompt_id="multi_field",
            )

        error_msg = str(context.exception)
        self.assertIn("field1", error_msg)
        self.assertIn("field2", error_msg)
        self.assertIn("field3", error_msg)


class TestWriteJsonlToS3(unittest.TestCase):
    """Test cases for write_jsonl_to_s3 following AAA pattern."""

    def setUp(self):
        """Set up test fixtures and patch module-level constants."""
        preprocess.wr.s3.to_parquet = lambda *args, **kwargs: None
        preprocess.wr.s3.read_parquet = lambda *args, **kwargs: []
        preprocess.wr.s3.list_objects = lambda *args, **kwargs: []
        self.context = Mock()

        # Patch BUCKET_NAME if it's None (not set from environment)
        if preprocess.BUCKET_NAME is None:
            self.bucket_patch = patch('preprocess.BUCKET_NAME', 'test-bucket')
            self.bucket_patch.start()
            self.addCleanup(self.bucket_patch.stop)

    @patch('preprocess.s3_client')
    def test_should_write_records_as_jsonl(self, mock_s3_client):
        """It should write records as JSONL format to S3."""
        # Arrange
        records = [
            {"id": 1, "text": "First record"},
            {"id": 2, "text": "Second record"},
            {"id": 3, "text": "Third record"},
        ]
        key = "test/output.jsonl"

        # Act
        result = write_jsonl_to_s3(records, key)

        # Assert
        self.assertEqual(result, f"s3://test-bucket/{key}")

        # Verify S3 client call
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        self.assertEqual(call_args[1]["Bucket"], "test-bucket")
        self.assertEqual(call_args[1]["Key"], key)

        # Verify JSONL format
        body = call_args[1]["Body"]
        lines = body.split("\n")
        self.assertEqual(len(lines), 3)
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            self.assertEqual(parsed["id"], i + 1)

    @patch('preprocess.s3_client')
    def test_should_handle_empty_records(self, mock_s3_client):
        """It should handle empty record list."""
        # Arrange
        records = []
        key = "empty/output.jsonl"

        # Act
        result = write_jsonl_to_s3(records, key)

        # Assert
        self.assertEqual(result, f"s3://test-bucket/{key}")
        mock_s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key=key,
            Body=""
        )

    @patch('preprocess.s3_client')
    def test_should_handle_complex_nested_records(self, mock_s3_client):
        """It should serialize complex nested structures."""
        # Arrange
        records = [
            {
                "recordId": "rec-1",
                "modelInput": {
                    "messages": [{"role": "user", "content": "test"}],
                    "params": {"temperature": 0.7},
                }
            }
        ]
        key = "complex/output.jsonl"

        # Act
        result = write_jsonl_to_s3(records, key)

        # Assert
        self.assertEqual(result, f"s3://test-bucket/{key}")
        call_args = mock_s3_client.put_object.call_args
        body = call_args[1]["Body"]
        parsed = json.loads(body)
        self.assertEqual(parsed["recordId"], "rec-1")
        self.assertEqual(parsed["modelInput"]["params"]["temperature"], 0.7)


class TestLambdaHandler(unittest.TestCase):
    """Test cases for lambda_handler following AAA pattern."""

    def setUp(self):
        """Set up test fixtures and patch module-level constants."""
        if not hasattr(preprocess.wr.s3, 'to_parquet'):
            preprocess.wr.s3.to_parquet = lambda *args, **kwargs: None
        if not hasattr(preprocess.wr.s3, 'read_parquet'):
            preprocess.wr.s3.read_parquet = lambda *args, **kwargs: []
        if not hasattr(preprocess.wr.s3, 'list_objects'):
            preprocess.wr.s3.list_objects = lambda *args, **kwargs: []
        self.context = Mock()

        # Patch BUCKET_NAME if it's None (not set from environment)
        if preprocess.BUCKET_NAME is None:
            self.bucket_patch = patch('preprocess.BUCKET_NAME', 'test-bucket')
            self.bucket_patch.start()
            self.addCleanup(self.bucket_patch.stop)

        # Common patches
        self.load_patch = mock.patch("preprocess.utils.load_files_in_chunks")
        self.write_patch = mock.patch(
            "preprocess.write_jsonl_to_s3", return_value="s3://bucket/key.jsonl"
        )
        self.parquet_patch = mock.patch("preprocess.wr.s3.to_parquet")
        self.uuid_patch = mock.patch(
            "preprocess.uuid4", side_effect=(f"id-{i}" for i in range(100000))
        )
        self.processor_patch = mock.patch("preprocess.get_processor_for_model_id")

    def tearDown(self):
        """Clean up patches."""
        mock.patch.stopall()

    def test_should_require_data_source(self):
        """It should raise error when neither s3_uri nor dataset_id is provided."""
        # Arrange
        event = {
            "job_name_prefix": "test",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        }

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            lambda_handler(event, self.context)

        self.assertIn("dataset_id", str(context.exception))
        self.assertIn("s3_uri", str(context.exception))

    def test_should_process_embedding_model_chunks(self):
        """It should process chunks for embedding models correctly."""
        # Arrange
        load_mock = self.load_patch.start()
        write_mock = self.write_patch.start()
        parquet_mock = self.parquet_patch.start()
        self.uuid_patch.start()  # Start but don't need to track
        processor_mock = self.processor_patch.start()

        # Setup mocks
        mock_processor = Mock()
        mock_processor.model_type = "embedding"
        mock_processor.process_input.return_value = {
            "recordId": "test-id",
            "modelInput": {"inputText": "test"}
        }
        processor_mock.return_value = mock_processor

        def _yield_chunks(*args, **kwargs):
            del args, kwargs  # Unused
            for i in range(2):
                yield (i, pd.DataFrame([{"input_text": f"text-{i}"}]))

        load_mock.side_effect = _yield_chunks

        event = {
            "s3_uri": "s3://source/input.csv",
            "job_name_prefix": "embed-test",
            "model_id": "amazon.titan-embed-text-v2:0",
        }

        # Act
        result = lambda_handler(event, self.context)

        # Assert
        self.assertEqual(len(result["jobs"]), 2)
        self.assertEqual(result["jobs"][0]["model_id"], "amazon.titan-embed-text-v2:0")
        write_mock.assert_called()
        parquet_mock.assert_called()

    @patch('preprocess.pt.prompt_id_to_template', {"test_prompt": "Tell me about {topic}"})
    def test_should_process_text_model_with_prompt(self):
        """It should process text models with prompt templates."""
        # Arrange
        load_mock = self.load_patch.start()
        self.write_patch.start()  # Start but don't need to track
        self.parquet_patch.start()  # Start but don't need to track
        self.uuid_patch.start()  # Start but don't need to track
        processor_mock = self.processor_patch.start()

        # Setup mocks
        mock_processor = Mock()
        mock_processor.model_type = "text"
        mock_processor.process_input.return_value = {
            "recordId": "test-id",
            "modelInput": {"messages": [{"role": "user", "content": "test"}]}
        }
        processor_mock.return_value = mock_processor

        def _yield_chunks(*args, **kwargs):
            del args, kwargs  # Unused
            yield (0, pd.DataFrame([{"topic": "AI", "record_id": "rec-1"}]))

        load_mock.side_effect = _yield_chunks

        event = {
            "s3_uri": "s3://source/input.csv",
            "job_name_prefix": "text-test",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "prompt_id": "test_prompt",
        }

        # Act
        result = lambda_handler(event, self.context)

        # Assert
        self.assertEqual(len(result["jobs"]), 1)
        processor_mock.assert_called_with("anthropic.claude-3-haiku-20240307-v1:0")

    def test_should_respect_max_records_total(self):
        """It should limit total records processed across all chunks."""
        # Arrange
        load_mock = self.load_patch.start()
        self.write_patch.start()  # Start but don't need to track
        self.parquet_patch.start()  # Start but don't need to track
        self.uuid_patch.start()  # Start but don't need to track
        processor_mock = self.processor_patch.start()

        # Setup mocks
        mock_processor = Mock()
        mock_processor.model_type = "embedding"
        mock_processor.process_input.return_value = {"recordId": "id", "modelInput": {}}
        processor_mock.return_value = mock_processor

        def _yield_chunks(*args, **kwargs):
            del args, kwargs  # Unused
            for i in range(10):
                yield (i, pd.DataFrame([{"input_text": f"text-{j}"} for j in range(1000)]))

        load_mock.side_effect = _yield_chunks

        event = {
            "s3_uri": "s3://source/input.csv",
            "job_name_prefix": "limit-test",
            "model_id": "amazon.titan-embed-text-v2:0",
            "max_records_total": 2500,  # Should stop after 3 chunks
        }

        # Act
        result = lambda_handler(event, self.context)

        # Assert
        self.assertEqual(len(result["jobs"]), 3)  # 2 full chunks + 1 partial

    def test_should_respect_max_num_jobs(self):
        """It should limit the number of jobs created."""
        # Arrange
        load_mock = self.load_patch.start()
        self.write_patch.start()  # Start but don't need to track
        self.parquet_patch.start()  # Start but don't need to track
        self.uuid_patch.start()  # Start but don't need to track
        processor_mock = self.processor_patch.start()

        # Setup mocks
        mock_processor = Mock()
        mock_processor.model_type = "embedding"
        mock_processor.process_input.return_value = {"recordId": "id", "modelInput": {}}
        processor_mock.return_value = mock_processor

        def _yield_chunks(*args, **kwargs):
            del args, kwargs  # Unused
            for i in range(10):
                yield (i, pd.DataFrame([{"input_text": f"text-{i}"}]))

        load_mock.side_effect = _yield_chunks

        event = {
            "s3_uri": "s3://source/input.csv",
            "job_name_prefix": "jobs-limit",
            "model_id": "amazon.titan-embed-text-v2:0",
            "max_num_jobs": 3,
        }

        # Act
        result = lambda_handler(event, self.context)

        # Assert
        self.assertEqual(len(result["jobs"]), 3)

    def test_should_add_record_ids_when_missing(self):
        """It should generate record IDs when not present in data."""
        # Arrange
        load_mock = self.load_patch.start()
        self.write_patch.start()  # Start but don't need to track
        self.parquet_patch.start()  # Start but don't need to track
        uuid_mock = self.uuid_patch.start()  # Track for assertion
        processor_mock = self.processor_patch.start()

        # Setup mocks
        mock_processor = Mock()
        mock_processor.model_type = "embedding"
        mock_processor.process_input.return_value = {"recordId": "id", "modelInput": {}}
        processor_mock.return_value = mock_processor

        # Create DataFrame without record_id column
        df = pd.DataFrame([{"input_text": "test"}])

        def _yield_chunks(*args, **kwargs):
            del args, kwargs  # Unused
            yield (0, df)

        load_mock.side_effect = _yield_chunks

        event = {
            "s3_uri": "s3://source/input.csv",
            "job_name_prefix": "id-test",
            "model_id": "amazon.titan-embed-text-v2:0",
        }

        # Act
        lambda_handler(event, self.context)

        # Assert
        # UUID should have been called to generate ID
        uuid_mock.assert_called()

    def test_should_handle_invalid_max_records_total(self):
        """It should raise error for invalid max_records_total."""
        # Arrange
        test_cases = [0, -1, -100]

        for invalid_value in test_cases:
            event = {
                "s3_uri": "s3://source/input.csv",
                "job_name_prefix": "test",
                "model_id": "amazon.titan-embed-text-v2:0",
                "max_records_total": invalid_value,
            }

            # Act & Assert
            with self.assertRaises(ValueError) as context:
                lambda_handler(event, self.context)

            self.assertIn("positive integer", str(context.exception))

    @patch('preprocess.load_dataset')
    def test_should_handle_huggingface_dataset(self, mock_load_dataset):
        """It should process Hugging Face datasets."""
        # Arrange
        self.write_patch.start()  # Start but don't need to track
        parquet_mock = self.parquet_patch.start()  # Track for assertion
        self.uuid_patch.start()  # Start but don't need to track
        processor_mock = self.processor_patch.start()

        # Setup mocks
        mock_processor = Mock()
        mock_processor.model_type = "embedding"
        mock_processor.process_input.return_value = {"recordId": "id", "modelInput": {}}
        processor_mock.return_value = mock_processor

        # Mock HF dataset
        mock_batch = {"input_text": ["text1", "text2"]}
        mock_dataset = Mock()
        mock_dataset.batch.return_value = [mock_batch]
        mock_load_dataset.return_value = mock_dataset

        event = {
            "dataset_id": "test/dataset",
            "split": "train",
            "job_name_prefix": "hf-test",
            "model_id": "amazon.titan-embed-text-v2:0",
        }

        # Act
        lambda_handler(event, self.context)

        # Assert
        mock_load_dataset.assert_called_with("test/dataset", split="train", streaming=True)
        parquet_mock.assert_called()  # Should save HF data to S3


# Backward compatibility tests from original file
class ResolvePromptConfigurationTests(unittest.TestCase):
    """Validate the prompt configuration helper (backward compatibility)."""

    def test_embedding_model_returns_empty_configuration(self) -> None:
        template, fields = resolve_prompt_configuration("embedding", {})
        self.assertIsNone(template)
        self.assertEqual(fields, set())

    def test_missing_prompt_id_raises(self) -> None:
        with self.assertRaises(ValueError) as err:
            resolve_prompt_configuration("text", {"prompt_id": None})
        self.assertIn("prompt_id", str(err.exception))

    def test_unknown_prompt_id_lists_available(self) -> None:
        with self.assertRaises(ValueError) as err:
            resolve_prompt_configuration("text", {"prompt_id": "does_not_exist"})
        self.assertIn("does_not_exist", str(err.exception))

    @patch('preprocess.pt.prompt_id_to_template', {"joke_about_topic": "Tell a joke about {topic}"})
    def test_known_prompt_returns_required_fields(self) -> None:
        template, fields = resolve_prompt_configuration(
            "text", {"prompt_id": "joke_about_topic"}
        )
        self.assertIn("{topic", template)
        self.assertEqual(fields, {"topic"})


class ValidateChunkColumnsTests(unittest.TestCase):
    """Check dataset column validation for embeddings and text prompts (backward compatibility)."""

    def test_embedding_requires_input_text(self) -> None:
        with self.assertRaises(ValueError) as err:
            validate_chunk_columns(
                "embedding",
                {"not_input_text"},
                required_prompt_fields=set(),
                job_name_prefix="demo",
                prompt_id=None,
            )
        self.assertIn("input_text", str(err.exception))

    def test_embedding_passes_with_input_text(self) -> None:
        validate_chunk_columns(
            "embedding",
            {"input_text", "record_id"},
            required_prompt_fields=set(),
            job_name_prefix="demo",
            prompt_id=None,
        )  # should not raise

    def test_text_missing_required_fields_raises(self) -> None:
        with self.assertRaises(ValueError) as err:
            validate_chunk_columns(
                "text",
                {"record_id"},
                required_prompt_fields={"topic"},
                job_name_prefix="demo",
                prompt_id="joke_about_topic",
            )
        self.assertIn("topic", str(err.exception))

    def test_text_with_required_fields_passes(self) -> None:
        validate_chunk_columns(
            "text",
            {"record_id", "topic"},
            required_prompt_fields={"topic"},
            job_name_prefix="demo",
            prompt_id="joke_about_topic",
        )  # should not raise


if __name__ == "__main__":
    unittest.main()
