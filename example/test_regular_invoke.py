#!/usr/bin/env python3
"""
Test script for Bedrock regular invoke-model as alternative to batch inference.
Use this to test small datasets while waiting for batch inference access.
"""

import json
import time
import boto3
import pandas as pd
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import sys

# Initialize Bedrock runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='eu-central-2')


def invoke_claude_model(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 500
) -> Dict[str, Any]:
    """Invoke Claude model with given messages."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        return {
            'success': True,
            'content': result.get('content', [{}])[0].get('text', ''),
            'usage': result.get('usage', {}),
            'stop_reason': result.get('stop_reason')
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'content': None
        }


def invoke_embedding_model(
    model_id: str,
    input_text: str
) -> Dict[str, Any]:
    """Invoke Titan embedding model."""

    body = {
        "inputText": input_text
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        return {
            'success': True,
            'embedding': result.get('embedding'),
            'inputTextTokenCount': result.get('inputTextTokenCount')
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'embedding': None
        }


def process_text_dataset(
    data: List[Dict],
    model_id: str,
    prompt_template: str,
    max_tokens: int = 500,
    rate_limit_delay: float = 0.1
) -> List[Dict]:
    """Process text data with Claude models."""

    results = []

    print(f"\nProcessing {len(data)} records with {model_id}")
    print(f"Rate limit delay: {rate_limit_delay}s between requests\n")

    for record in tqdm(data, desc="Processing"):
        # Format the prompt with data from record
        try:
            prompt = prompt_template.format(**record)
        except KeyError as e:
            print(f"Warning: Missing field {e} in record, skipping...")
            continue

        messages = [{"role": "user", "content": prompt}]

        # Invoke the model
        result = invoke_claude_model(model_id, messages, max_tokens)

        # Combine input record with output
        output_record = record.copy()
        output_record['response'] = result.get('content', '')
        output_record['success'] = result.get('success', False)
        if not result['success']:
            output_record['error'] = result.get('error', '')

        results.append(output_record)

        # Rate limiting
        time.sleep(rate_limit_delay)

    return results


def process_embedding_dataset(
    data: List[Dict],
    model_id: str,
    rate_limit_delay: float = 0.05
) -> List[Dict]:
    """Process embedding data with Titan model."""

    results = []

    print(f"\nProcessing {len(data)} records with {model_id}")
    print(f"Rate limit delay: {rate_limit_delay}s between requests\n")

    for record in tqdm(data, desc="Processing embeddings"):
        input_text = record.get('input_text', '')

        if not input_text:
            print("Warning: Empty input_text, skipping...")
            continue

        # Invoke the model
        result = invoke_embedding_model(model_id, input_text)

        # Combine input record with output
        output_record = record.copy()
        output_record['embedding'] = result.get('embedding', [])
        output_record['token_count'] = result.get('inputTextTokenCount', 0)
        output_record['success'] = result.get('success', False)
        if not result['success']:
            output_record['error'] = result.get('error', '')

        results.append(output_record)

        # Rate limiting (embeddings are faster)
        time.sleep(rate_limit_delay)

    return results


def load_sample_data(file_path: str = None) -> List[Dict]:
    """Load data from CSV or create sample data."""

    if file_path:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    else:
        # Sample data for testing
        return [
            {"topic": "quantum computing", "source": "What is quantum computing?"},
            {"topic": "machine learning", "source": "Explain neural networks"},
            {"topic": "climate change", "source": "What causes global warming?"},
            {"topic": "space exploration", "source": "Why explore Mars?"},
            {"topic": "renewable energy", "source": "Benefits of solar power"}
        ]


def save_results(results: List[Dict], output_file: str):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


# Prompt templates (matching your project)
PROMPT_TEMPLATES = {
    "joke_about_topic": "Tell me a joke about {topic} in less than 50 words.",
    "question_answering": "Answer this question concisely:\n\nQuestion: {source}\n\nAnswer:",
    "summarization": "Summarize the following text in 2-3 sentences:\n\n{text}",
    "sentiment": "Analyze the sentiment of this text (positive/negative/neutral):\n\n{text}"
}


def main():
    parser = argparse.ArgumentParser(description='Test Bedrock models with regular invoke')
    parser.add_argument('--model-id', default='anthropic.claude-3-haiku-20240307-v1:0',
                        help='Model ID to use')
    parser.add_argument('--model-type', choices=['text', 'embedding'], default='text',
                        help='Type of model')
    parser.add_argument('--input-file', help='Input CSV file path')
    parser.add_argument('--output-file', default='test_output.csv',
                        help='Output CSV file path')
    parser.add_argument('--prompt-id', default='question_answering',
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help='Prompt template to use')
    parser.add_argument('--max-records', type=int, default=10,
                        help='Maximum number of records to process')
    parser.add_argument('--rate-limit-delay', type=float, default=0.1,
                        help='Delay between API calls in seconds')
    parser.add_argument('--max-tokens', type=int, default=500,
                        help='Max tokens for response (text models only)')

    args = parser.parse_args()

    # Load data
    data = load_sample_data(args.input_file)

    # Limit records for testing
    if args.max_records and len(data) > args.max_records:
        data = data[:args.max_records]
        print(f"Limiting to {args.max_records} records for testing")

    # Process based on model type
    if args.model_type == 'text':
        prompt_template = PROMPT_TEMPLATES[args.prompt_id]
        print(f"Using prompt template: {args.prompt_id}")
        results = process_text_dataset(
            data,
            args.model_id,
            prompt_template,
            args.max_tokens,
            args.rate_limit_delay
        )
    else:
        # For embeddings, ensure data has 'input_text' column
        if not all('input_text' in record for record in data):
            # If no input_text, try to use first text field
            for record in data:
                if 'input_text' not in record:
                    record['input_text'] = record.get('text', '') or record.get('source', '') or str(record.get('topic', ''))

        results = process_embedding_dataset(
            data,
            args.model_id,
            args.rate_limit_delay
        )

    # Calculate statistics
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful

    print(f"\n=== Results ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Save results
    save_results(results, args.output_file)

    # Show sample output
    if results and successful > 0:
        print("\n=== Sample Output ===")
        sample = next(r for r in results if r.get('success', False))
        if args.model_type == 'text':
            print(f"Input: {sample.get('source', sample.get('topic', 'N/A'))}")
            print(f"Response: {sample.get('response', 'N/A')[:200]}...")
        else:
            print(f"Input: {sample.get('input_text', 'N/A')[:100]}...")
            print(f"Embedding dimensions: {len(sample.get('embedding', []))}")
            print(f"Token count: {sample.get('token_count', 'N/A')}")


if __name__ == "__main__":
    main()