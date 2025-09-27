"""Model-specific processors for Amazon Bedrock batch inference.

This module provides an abstraction layer for handling different AI model providers
in the Bedrock ecosystem. Each model provider (Anthropic, Amazon Titan, etc.) has
unique API requirements for input formatting and output parsing.

The processor pattern allows for:
- Model-specific input formatting (Messages API, embedding format, etc.)
- Standardized output parsing across different response structures
- Easy extension for new model providers
- Type-safe batch inference record handling

To add support for a new model:
1. Create a new class extending BaseProcessor
2. Implement process_input() to format requests per the model's API
3. Implement process_output() to parse the model's response format
4. Update get_processor_for_model_id() to map model IDs to your processor

Example:
    >>> processor = get_processor_for_model_id("anthropic.claude-3-haiku-20240307-v1:0")
    >>> input_record = processor.process_input("Hello, world!", "record-123")
    >>> # After Bedrock processes the batch...
    >>> output = processor.process_output(bedrock_response)
"""

from abc import ABC, abstractmethod
from typing import Literal

from custom_types import BatchInferenceRecord


class BaseProcessor(ABC):
    """Abstract base class for Bedrock model input/output processing.

    This class defines the interface that all model-specific processors must implement.
    Each processor handles the unique requirements of its model provider's API,
    ensuring correct formatting of batch inference requests and parsing of responses.

    Attributes:
        model_type: Categorizes the model as either 'embedding' (for vector generation)
            or 'text' (for language generation). This affects how data flows through
            the preprocessing pipeline.

    Note:
        Subclasses must set the model_type class attribute and implement both
        abstract methods to handle their specific model's API requirements.
    """

    # Model category: determines preprocessing behavior and validation requirements
    model_type: Literal["embedding", "text"]

    @abstractmethod
    def process_input(
        self, input_text: str, record_id: str, **kwargs
    ) -> BatchInferenceRecord:
        """Format input text into a Bedrock batch inference record.

        Transforms raw text into the specific JSON structure required by the model's API.
        Each model provider has unique requirements for request formatting.

        Args:
            input_text: The text to process. For text models, this is typically
                a prompt or question. For embedding models, it's the text to vectorize.
            record_id: Unique identifier for tracking this record through the pipeline.
                Used to correlate inputs with outputs in batch processing.
            **kwargs: Additional model-specific parameters (e.g., max_tokens, temperature).

        Returns:
            BatchInferenceRecord: Dictionary with 'recordId' and 'modelInput' keys,
                formatted according to Bedrock's batch API requirements.

        Note:
            The returned structure must conform to Bedrock's JSONL format:
            {"recordId": "...", "modelInput": {...}}
        """
        pass

    @abstractmethod
    def process_output(self, output_data: dict, **kwargs) -> dict:
        """Parse and extract relevant data from model output.

        Processes the raw response from Bedrock batch inference into a standardized
        format for downstream consumption. Each model returns data in its own
        structure that needs to be normalized.

        Args:
            output_data: Raw output from Bedrock batch inference containing:
                - recordId: The same ID used in the input
                - modelOutput: Model-specific response structure
            **kwargs: Additional processing options (currently unused but reserved
                for future extensibility).

        Returns:
            Dictionary with standardized keys:
                - record_id: Original record identifier for correlation
                - response (for text models): Generated text content
                - embedding (for embedding models): Vector representation

        Note:
            The output structure varies by model type but always includes
            the record_id for joining with original input data.
        """
        pass


class AnthropicProcessor(BaseProcessor):
    """Processor for Anthropic Claude models using the Messages API.

    Handles all Claude model variants (Haiku, Sonnet, Opus) by formatting
    requests according to Anthropic's Messages API specification. This API
    uses a conversation format with role-based messages.

    Supported models:
        - anthropic.claude-3-haiku-*
        - anthropic.claude-3-sonnet-*
        - anthropic.claude-3-opus-*
        - anthropic.claude-3-5-sonnet-*

    The Messages API requires:
        - anthropic_version: API version identifier
        - max_tokens: Maximum response length
        - messages: Array of conversation turns with roles and content
    """

    model_type = "text"  # Claude models generate text responses

    def process_input(
        self, input_text: str, record_id: str, **kwargs
    ) -> BatchInferenceRecord:
        """Format input for Anthropic's Messages API.

        Creates a single-turn conversation with the user's input text.
        The Messages API supports multi-modal content, but this implementation
        focuses on text-only interactions for batch processing.

        Args:
            input_text: The prompt or question to send to Claude.
            record_id: Unique identifier for batch processing correlation.
            **kwargs: Optional parameters:
                - max_tokens: Maximum response length (default: 1024)
                - temperature: Randomness in generation (0-1)
                - top_p: Nucleus sampling parameter

        Returns:
            BatchInferenceRecord formatted for Claude's Messages API.

        Example:
            >>> processor = AnthropicProcessor()
            >>> record = processor.process_input(
            ...     "What is machine learning?",
            ...     "rec-001",
            ...     max_tokens=500
            ... )
        """
        return {
            "recordId": record_id,
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",  # Required API version
                "max_tokens": kwargs.get("max_tokens", 1024),  # Response length limit
                "messages": [
                    {
                        "role": "user",  # Single user message for batch inference
                        "content": [
                            {
                                "type": "text",  # Text-only content for batch
                                "text": input_text,
                            }
                        ],
                    }
                ],
            },
        }

    def process_output(self, output_data: dict, **kwargs) -> dict:
        """Extract text response from Claude's output.

        Parses the Messages API response structure to extract the generated text.
        Claude returns content as an array of content blocks; we extract the
        text from the last (typically only) content block.

        Args:
            output_data: Raw Bedrock response containing:
                - recordId: Original record identifier
                - modelOutput: Claude's response with content array
            **kwargs: Currently unused, reserved for future options.

        Returns:
            Dictionary with:
                - record_id: Original identifier for correlation
                - response: Generated text from Claude

        Note:
            Handles missing or malformed responses gracefully by providing
            default structure to avoid KeyError exceptions.
        """
        # Safely extract model output with fallback for error cases
        model_output = output_data.get("modelOutput", {"content": [{"text": None}]})
        return {
            "record_id": output_data["recordId"],
            "response": model_output["content"][-1][
                "text"
            ],  # Last content block's text
        }


class TitanV2Processor(BaseProcessor):
    """Processor for Amazon Titan Text Embeddings V2 model.

    Handles the Titan embedding model that converts text into dense vector
    representations. These embeddings are useful for semantic search,
    clustering, and similarity comparisons.

    Model ID: amazon.titan-embed-text-v2:0

    The Titan V2 embedding model:
        - Generates 1024-dimensional vectors by default
        - Supports dimension reduction (256, 512, 1024)
        - Handles up to 8,192 tokens of input text
        - Optimized for semantic similarity tasks
    """

    model_type = "embedding"  # Generates vector representations

    def process_input(
        self, input_text: str, record_id: str, **kwargs
    ) -> BatchInferenceRecord:
        """Format input for Titan V2 embedding generation.

        Creates a simple request structure with the text to embed.
        Titan's embedding API has minimal requirements compared to text models.

        Args:
            input_text: Text to convert into vector representation.
                Should be meaningful content for best embedding quality.
            record_id: Unique identifier for batch processing correlation.
            **kwargs: Optional parameters:
                - dimensions: Output vector size (256, 512, or 1024)
                - normalize: Whether to L2-normalize the embedding

        Returns:
            BatchInferenceRecord formatted for Titan embedding API.

        Example:
            >>> processor = TitanV2Processor()
            >>> record = processor.process_input(
            ...     "Machine learning is a subset of artificial intelligence.",
            ...     "rec-002",
            ...     dimensions=512
            ... )
        """
        return {
            "recordId": record_id,
            "modelInput": {
                "inputText": input_text,  # Text to encode as embedding
                **kwargs,  # Pass through any additional parameters
            },
        }

    def process_output(self, output_data: dict, **kwargs) -> dict:
        """Extract embedding vector from Titan's output.

        Parses the response to retrieve the dense vector representation
        of the input text. The embedding is a list of floating-point numbers
        representing the text in high-dimensional space.

        Args:
            output_data: Raw Bedrock response containing:
                - recordId: Original record identifier
                - modelOutput: Dictionary with 'embedding' key containing vector
            **kwargs: Currently unused, reserved for future options.

        Returns:
            Dictionary with:
                - record_id: Original identifier for correlation
                - embedding: List of floats representing the text vector

        Note:
            The embedding dimension depends on the model configuration
            (typically 1024 for Titan V2 unless specified otherwise).
        """
        return {
            "record_id": output_data["recordId"],
            "embedding": output_data["modelOutput"][
                "embedding"
            ],  # Dense vector representation
        }


def get_processor_for_model_id(model_id: str) -> BaseProcessor:
    """Factory function to instantiate the appropriate processor for a model.

    Maps Bedrock model IDs to their corresponding processor implementations.
    This function serves as the central registry for supported models and
    should be updated when adding support for new model providers.

    Args:
        model_id: Bedrock model identifier string. Examples:
            - "anthropic.claude-3-haiku-20240307-v1:0"
            - "anthropic.claude-3-5-sonnet-20241022-v2:0"
            - "amazon.titan-embed-text-v2:0"

    Returns:
        BaseProcessor: Instance of the appropriate processor for the model.

    Raises:
        ValueError: If the model_id is not recognized or supported.

    Note:
        Model IDs in Bedrock follow the pattern: provider.model-name-version:revision
        We use substring matching to handle version variations within model families.

    Example:
        >>> processor = get_processor_for_model_id("anthropic.claude-3-haiku-20240307-v1:0")
        >>> isinstance(processor, AnthropicProcessor)
        True
    """
    # Check for Anthropic Claude models (all variants)
    if "anthropic" in model_id:
        return AnthropicProcessor()
    # Check for Amazon Titan V2 embedding model
    elif "amazon.titan-embed-text-v2:0" in model_id:
        return TitanV2Processor()
    # Model not supported - provide helpful error message
    else:
        raise ValueError(
            f"Unsupported model_id: {model_id}. Only Anthropic and Titan V2 embeddings are supported."
        )
