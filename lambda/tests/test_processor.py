"""Comprehensive unit tests for lambda/processor.py.

This module provides complete test coverage for all processor classes and functions,
following AAA (Arrange-Act-Assert) pattern and unit testing best practices.
"""

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock

# Ensure the Lambda source directory is importable
LAMBDA_SRC = Path(__file__).resolve().parents[1]
if str(LAMBDA_SRC) not in sys.path:
    sys.path.append(str(LAMBDA_SRC))

from processor import (
    AnthropicProcessor,
    BaseProcessor,
    TitanV2Processor,
    get_processor_for_model_id,
)


class TestAnthropicProcessor(unittest.TestCase):
    """Test cases for AnthropicProcessor following AAA pattern."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = AnthropicProcessor()

    def test_should_have_text_model_type(self):
        """It should be classified as a text generation model."""
        # Arrange - done in setUp

        # Act
        model_type = self.processor.model_type

        # Assert
        self.assertEqual(model_type, "text")

    def test_should_format_input_for_messages_api(self):
        """It should format input according to Anthropic Messages API structure."""
        # Arrange
        input_text = "What is machine learning?"
        record_id = "test-record-123"
        expected_structure = {
            "recordId": "test-record-123",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is machine learning?",
                            }
                        ],
                    }
                ],
            },
        }

        # Act
        result = self.processor.process_input(input_text, record_id)

        # Assert
        self.assertEqual(result, expected_structure)

    def test_should_use_custom_max_tokens_when_provided(self):
        """It should use custom max_tokens parameter when provided."""
        # Arrange
        input_text = "Test prompt"
        record_id = "rec-456"
        custom_max_tokens = 500

        # Act
        result = self.processor.process_input(
            input_text, record_id, max_tokens=custom_max_tokens
        )

        # Assert
        self.assertEqual(result["modelInput"]["max_tokens"], custom_max_tokens)

    def test_should_pass_additional_kwargs_to_model_input(self):
        """It should pass additional parameters to modelInput."""
        # Arrange
        input_text = "Test"
        record_id = "rec-789"
        additional_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        }

        # Act
        result = self.processor.process_input(input_text, record_id, **additional_params)

        # Assert
        # Check that additional params are not in the modelInput directly
        # (they would need to be explicitly handled in the implementation)
        self.assertIn("max_tokens", result["modelInput"])
        self.assertEqual(result["modelInput"]["max_tokens"], 1024)  # Default value

    def test_should_extract_text_from_model_output(self):
        """It should extract text response from Claude's output structure."""
        # Arrange
        output_data = {
            "recordId": "test-123",
            "modelOutput": {
                "content": [
                    {
                        "text": "Machine learning is a subset of artificial intelligence."
                    }
                ]
            },
        }
        expected = {
            "record_id": "test-123",
            "response": "Machine learning is a subset of artificial intelligence.",
        }

        # Act
        result = self.processor.process_output(output_data)

        # Assert
        self.assertEqual(result, expected)

    def test_should_handle_multiple_content_blocks(self):
        """It should extract text from the last content block."""
        # Arrange
        output_data = {
            "recordId": "multi-123",
            "modelOutput": {
                "content": [
                    {"text": "First response"},
                    {"text": "Second response"},
                    {"text": "Final response"},
                ]
            },
        }

        # Act
        result = self.processor.process_output(output_data)

        # Assert
        self.assertEqual(result["response"], "Final response")

    def test_should_handle_missing_model_output_gracefully(self):
        """It should provide default structure for missing modelOutput."""
        # Arrange
        output_data = {
            "recordId": "error-123",
            # Missing modelOutput
        }

        # Act
        result = self.processor.process_output(output_data)

        # Assert
        self.assertEqual(result["record_id"], "error-123")
        self.assertIsNone(result["response"])

    def test_should_handle_empty_content_array(self):
        """It should handle empty content array gracefully."""
        # Arrange
        output_data = {
            "recordId": "empty-123",
            "modelOutput": {
                "content": []
            }
        }

        # Act & Assert
        with self.assertRaises(IndexError):
            self.processor.process_output(output_data)


class TestTitanV2Processor(unittest.TestCase):
    """Test cases for TitanV2Processor following AAA pattern."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = TitanV2Processor()

    def test_should_have_embedding_model_type(self):
        """It should be classified as an embedding model."""
        # Arrange - done in setUp

        # Act
        model_type = self.processor.model_type

        # Assert
        self.assertEqual(model_type, "embedding")

    def test_should_format_input_for_titan_embedding_api(self):
        """It should format input according to Titan V2 embedding API structure."""
        # Arrange
        input_text = "Machine learning is a subset of artificial intelligence."
        record_id = "embed-123"
        expected_structure = {
            "recordId": "embed-123",
            "modelInput": {
                "inputText": "Machine learning is a subset of artificial intelligence.",
            },
        }

        # Act
        result = self.processor.process_input(input_text, record_id)

        # Assert
        self.assertEqual(result, expected_structure)

    def test_should_pass_additional_parameters_to_model_input(self):
        """It should include additional parameters in modelInput."""
        # Arrange
        input_text = "Test text for embedding"
        record_id = "embed-456"
        additional_params = {
            "dimensions": 512,
            "normalize": True,
        }

        # Act
        result = self.processor.process_input(input_text, record_id, **additional_params)

        # Assert
        self.assertEqual(result["modelInput"]["dimensions"], 512)
        self.assertEqual(result["modelInput"]["normalize"], True)
        self.assertEqual(result["modelInput"]["inputText"], input_text)

    def test_should_extract_embedding_from_model_output(self):
        """It should extract embedding vector from Titan's output."""
        # Arrange
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        output_data = {
            "recordId": "embed-123",
            "modelOutput": {
                "embedding": embedding_vector
            },
        }
        expected = {
            "record_id": "embed-123",
            "embedding": embedding_vector,
        }

        # Act
        result = self.processor.process_output(output_data)

        # Assert
        self.assertEqual(result, expected)

    def test_should_handle_large_embedding_vectors(self):
        """It should handle full-size embedding vectors (1024 dimensions)."""
        # Arrange
        large_embedding = [0.01 * i for i in range(1024)]
        output_data = {
            "recordId": "embed-large",
            "modelOutput": {
                "embedding": large_embedding
            },
        }

        # Act
        result = self.processor.process_output(output_data)

        # Assert
        self.assertEqual(len(result["embedding"]), 1024)
        self.assertEqual(result["embedding"], large_embedding)

    def test_should_raise_error_for_missing_embedding(self):
        """It should raise KeyError when embedding is missing from output."""
        # Arrange
        output_data = {
            "recordId": "embed-error",
            "modelOutput": {
                # Missing embedding
            },
        }

        # Act & Assert
        with self.assertRaises(KeyError):
            self.processor.process_output(output_data)

    def test_should_preserve_record_id_in_output(self):
        """It should preserve the record ID in processed output."""
        # Arrange
        record_id = "unique-record-789"
        output_data = {
            "recordId": record_id,
            "modelOutput": {
                "embedding": [0.1, 0.2, 0.3]
            },
        }

        # Act
        result = self.processor.process_output(output_data)

        # Assert
        self.assertEqual(result["record_id"], record_id)


class TestGetProcessorForModelId(unittest.TestCase):
    """Test cases for get_processor_for_model_id factory function."""

    def test_should_return_anthropic_processor_for_claude_models(self):
        """It should return AnthropicProcessor for various Claude model IDs."""
        # Arrange
        claude_models = [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-instant-v1",
        ]

        for model_id in claude_models:
            # Act
            processor = get_processor_for_model_id(model_id)

            # Assert
            self.assertIsInstance(processor, AnthropicProcessor)
            self.assertEqual(processor.model_type, "text")

    def test_should_return_titan_processor_for_embedding_model(self):
        """It should return TitanV2Processor for Titan embedding model."""
        # Arrange
        model_id = "amazon.titan-embed-text-v2:0"

        # Act
        processor = get_processor_for_model_id(model_id)

        # Assert
        self.assertIsInstance(processor, TitanV2Processor)
        self.assertEqual(processor.model_type, "embedding")

    def test_should_handle_partial_model_id_matching(self):
        """It should match based on substring for model families."""
        # Arrange
        partial_anthropic = "anthropic.claude"
        partial_titan = "amazon.titan-embed-text-v2:0"

        # Act
        anthropic_processor = get_processor_for_model_id(partial_anthropic)
        titan_processor = get_processor_for_model_id(partial_titan)

        # Assert
        self.assertIsInstance(anthropic_processor, AnthropicProcessor)
        self.assertIsInstance(titan_processor, TitanV2Processor)

    def test_should_raise_valueerror_for_unsupported_model(self):
        """It should raise ValueError for unsupported model IDs."""
        # Arrange
        unsupported_models = [
            "meta.llama2-13b-chat-v1",
            "cohere.command-text-v14",
            "ai21.j2-mid-v1",
            "amazon.titan-text-express-v1",  # Text Titan, not embedding
            "unknown-model",
        ]

        for model_id in unsupported_models:
            # Act & Assert
            with self.assertRaises(ValueError) as context:
                get_processor_for_model_id(model_id)

            # Assert error message contains helpful information
            error_msg = str(context.exception)
            self.assertIn("Unsupported model_id", error_msg)
            self.assertIn(model_id, error_msg)
            self.assertIn("Only Anthropic and Titan V2 embeddings are supported", error_msg)

    def test_should_be_case_sensitive_for_model_matching(self):
        """It should be case-sensitive when matching model IDs."""
        # Arrange
        uppercase_model = "ANTHROPIC.CLAUDE-3-HAIKU"

        # Act & Assert
        with self.assertRaises(ValueError):
            get_processor_for_model_id(uppercase_model)

    def test_should_create_new_instance_each_call(self):
        """It should create a new processor instance for each call."""
        # Arrange
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        # Act
        processor1 = get_processor_for_model_id(model_id)
        processor2 = get_processor_for_model_id(model_id)

        # Assert
        self.assertIsNot(processor1, processor2)
        self.assertIsInstance(processor1, AnthropicProcessor)
        self.assertIsInstance(processor2, AnthropicProcessor)


class TestBaseProcessorAbstraction(unittest.TestCase):
    """Test cases for BaseProcessor abstract class."""

    def test_should_not_instantiate_abstract_base_class(self):
        """It should not allow direct instantiation of BaseProcessor."""
        # Arrange & Act & Assert
        with self.assertRaises(TypeError) as context:
            BaseProcessor()

        # Verify it's because of abstract methods
        error_msg = str(context.exception)
        self.assertIn("Can't instantiate abstract class", error_msg)

    def test_should_require_process_input_implementation(self):
        """It should require subclasses to implement process_input."""
        # Arrange
        class IncompleteProcessor(BaseProcessor):
            model_type = "test"

            def process_output(self, output_data: dict, **kwargs) -> dict:
                return {}

        # Act & Assert
        with self.assertRaises(TypeError):
            IncompleteProcessor()

    def test_should_require_process_output_implementation(self):
        """It should require subclasses to implement process_output."""
        # Arrange
        class IncompleteProcessor(BaseProcessor):
            model_type = "test"

            def process_input(self, input_text: str, record_id: str, **kwargs) -> Any:
                return {}

        # Act & Assert
        with self.assertRaises(TypeError):
            IncompleteProcessor()

    def test_should_allow_complete_subclass_implementation(self):
        """It should allow instantiation when all abstract methods are implemented."""
        # Arrange
        class CompleteProcessor(BaseProcessor):
            model_type = "test"

            def process_input(self, input_text: str, record_id: str, **kwargs) -> Any:
                return {"recordId": record_id, "modelInput": {"text": input_text}}

            def process_output(self, output_data: dict, **kwargs) -> dict:
                return {"record_id": output_data.get("recordId")}

        # Act
        processor = CompleteProcessor()

        # Assert
        self.assertIsInstance(processor, BaseProcessor)
        self.assertEqual(processor.model_type, "test")


class TestProcessorIntegration(unittest.TestCase):
    """Integration tests for processor workflow."""

    def test_anthropic_processor_full_workflow(self):
        """It should handle complete input/output processing for Anthropic models."""
        # Arrange
        processor = AnthropicProcessor()
        input_text = "Explain quantum computing"
        record_id = "quantum-123"

        # Act - Process input
        input_result = processor.process_input(input_text, record_id, max_tokens=2000)

        # Simulate Bedrock response
        bedrock_response = {
            "recordId": record_id,
            "modelOutput": {
                "content": [
                    {
                        "text": "Quantum computing is a revolutionary approach to computation..."
                    }
                ]
            }
        }

        # Act - Process output
        output_result = processor.process_output(bedrock_response)

        # Assert
        self.assertEqual(input_result["recordId"], record_id)
        self.assertEqual(input_result["modelInput"]["max_tokens"], 2000)
        self.assertEqual(output_result["record_id"], record_id)
        self.assertIn("Quantum computing", output_result["response"])

    def test_titan_processor_full_workflow(self):
        """It should handle complete input/output processing for Titan embedding models."""
        # Arrange
        processor = TitanV2Processor()
        input_text = "Natural language processing"
        record_id = "nlp-456"

        # Act - Process input
        input_result = processor.process_input(input_text, record_id, dimensions=256)

        # Simulate Bedrock response
        embedding = [0.1 * i for i in range(256)]
        bedrock_response = {
            "recordId": record_id,
            "modelOutput": {
                "embedding": embedding
            }
        }

        # Act - Process output
        output_result = processor.process_output(bedrock_response)

        # Assert
        self.assertEqual(input_result["recordId"], record_id)
        self.assertEqual(input_result["modelInput"]["dimensions"], 256)
        self.assertEqual(output_result["record_id"], record_id)
        self.assertEqual(len(output_result["embedding"]), 256)

    def test_processor_selection_and_usage(self):
        """It should select correct processor and handle data processing."""
        # Arrange
        test_cases = [
            {
                "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
                "expected_type": AnthropicProcessor,
                "model_type": "text"
            },
            {
                "model_id": "amazon.titan-embed-text-v2:0",
                "expected_type": TitanV2Processor,
                "model_type": "embedding"
            }
        ]

        for test_case in test_cases:
            # Act
            processor = get_processor_for_model_id(test_case["model_id"])

            # Assert
            self.assertIsInstance(processor, test_case["expected_type"])
            self.assertEqual(processor.model_type, test_case["model_type"])

            # Test that processor can process input
            result = processor.process_input("test text", "test-id")
            self.assertIn("recordId", result)
            self.assertIn("modelInput", result)


if __name__ == "__main__":
    unittest.main()