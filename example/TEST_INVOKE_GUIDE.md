# Testing with Regular Invoke-Model

This guide shows how to test Bedrock models using regular `invoke-model` API calls instead of batch inference jobs.

## Quick Start

### 1. Basic Test (No Input File)

Test with built-in sample data:

```bash
python test_regular_invoke.py
```

### 2. Test with Your CSV File

```bash
python test_regular_invoke.py \
    --input-file sample_data.csv \
    --output-file results.csv \
    --max-records 5
```

### 3. Test Claude 3.5 Sonnet

```bash
python test_regular_invoke.py \
    --model-id anthropic.claude-3-5-sonnet-20240620-v1:0 \
    --input-file sample_data.csv \
    --prompt-id question_answering \
    --max-tokens 100 \
    --rate-limit-delay 0.5
```

### 4. Test Embeddings

```bash
# Create embedding input file
cat > embedding_input.csv << EOF
input_text
"Quantum computing uses quantum mechanics principles"
"Machine learning is a subset of artificial intelligence"
"Solar panels convert sunlight into electricity"
EOF

python test_regular_invoke.py \
    --model-type embedding \
    --model-id amazon.titan-embed-text-v2:0 \
    --input-file embedding_input.csv \
    --output-file embeddings.csv
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-id` | Bedrock model ID | `anthropic.claude-3-haiku-20240307-v1:0` |
| `--model-type` | `text` or `embedding` | `text` |
| `--input-file` | Input CSV file path | None (uses sample data) |
| `--output-file` | Output CSV file path | `test_output.csv` |
| `--prompt-id` | Prompt template name | `question_answering` |
| `--max-records` | Max records to process | 10 |
| `--rate-limit-delay` | Seconds between API calls | 0.1 |
| `--max-tokens` | Max response tokens | 500 |

## Available Prompt Templates

- `joke_about_topic`: Creates jokes about the topic
- `question_answering`: Answers questions (default)
- `summarization`: Summarizes text
- `sentiment`: Analyzes sentiment

## Input File Format

### For Text Models

CSV with columns matching your prompt template:

```csv
topic,source
quantum computing,What is quantum computing?
machine learning,Explain neural networks
```

### For Embedding Models

CSV with `input_text` column:

```csv
input_text
"First text to embed"
"Second text to embed"
```

## Example Workflows

### Test Small Dataset End-to-End

```bash
# 1. Test with 3 records first
python test_regular_invoke.py \
    --input-file sample_data.csv \
    --max-records 3 \
    --output-file test_3_records.csv

# 2. Check results
cat test_3_records.csv

# 3. If successful, test with more records
python test_regular_invoke.py \
    --input-file sample_data.csv \
    --max-records 10 \
    --output-file test_10_records.csv
```

### Compare Different Models

```bash
# Test with Haiku (faster, cheaper)
python test_regular_invoke.py \
    --model-id anthropic.claude-3-haiku-20240307-v1:0 \
    --input-file sample_data.csv \
    --output-file haiku_results.csv \
    --max-records 5

# Test with Sonnet (better quality)
python test_regular_invoke.py \
    --model-id anthropic.claude-3-5-sonnet-20240620-v1:0 \
    --input-file sample_data.csv \
    --output-file sonnet_results.csv \
    --max-records 5
```

### Test Different Prompts

```bash
# Test joke generation
python test_regular_invoke.py \
    --prompt-id joke_about_topic \
    --input-file sample_data.csv \
    --output-file jokes.csv

# Test sentiment analysis
python test_regular_invoke.py \
    --prompt-id sentiment \
    --input-file sample_data.csv \
    --output-file sentiment.csv
```

## Rate Limiting Considerations

- **Claude 3 Haiku**: ~10 requests/second → use `--rate-limit-delay 0.1`
- **Claude 3.5 Sonnet**: ~5 requests/second → use `--rate-limit-delay 0.2`
- **Titan Embeddings**: ~20 requests/second → use `--rate-limit-delay 0.05`

Adjust based on your account limits and quota.

## Cost Estimation

### Text Models (per 1000 input/output tokens)

- **Claude 3 Haiku**: $0.00025 / $0.00125
- **Claude 3.5 Sonnet**: $0.003 / $0.015

### Embedding Models

- **Titan Embeddings V2**: $0.00002 per 1000 tokens

### Example Costs

Processing 100 records with ~100 tokens input, ~200 tokens output:

- Haiku: ~$0.04
- Sonnet: ~$0.33
- Embeddings: ~$0.002

## Monitoring Progress

The script shows:

- Progress bar during processing
- Success/failure count
- Sample output
- Saved results location

## Troubleshooting

### "Access Denied" Error

```bash
# Verify model access
aws bedrock get-foundation-model-availability \
    --model-id anthropic.claude-3-haiku-20240307-v1:0 \
    --region eu-central-2
```

### "Rate Exceeded" Error

Increase `--rate-limit-delay`:

```bash
python test_regular_invoke.py --rate-limit-delay 1.0
```

### Missing Fields in CSV

Ensure your CSV has columns matching the prompt template variables.

## Next Steps

Once you've verified your models work:

1. **Request Batch Inference Access**: Follow instructions in `BATCH_INFERENCE_ACCESS.md`
2. **Prepare Full Dataset**: Format your complete dataset matching tested structure
3. **Deploy CDK Stack**: When batch access is granted, deploy the full solution
4. **Run Batch Jobs**: Process large datasets at 50% discount

## Python Dependencies

Install required packages:

```bash
pip install boto3 pandas tqdm
```

Or use the existing Lambda requirements:

```bash
pip install -r lambda/requirements.txt
```
