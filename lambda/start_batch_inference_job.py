"""Lambda function for initiating Bedrock batch inference jobs.

This module serves as the bridge between Step Functions orchestration and
Bedrock's batch inference API. It handles job submission, state tracking,
and integration with the distributed task management system.

Key responsibilities:
- Submit batch inference jobs to Amazon Bedrock
- Store job metadata and task tokens in DynamoDB for tracking
- Handle API throttling with aggressive retry configuration
- Validate job submission and capture initial status
- Enable asynchronous job completion notifications via task tokens

The function operates within a Step Functions Map state that controls
concurrency, ensuring the system doesn't exceed Bedrock's job limits
or overwhelm the API with simultaneous requests.

Task token pattern:
This Lambda uses Step Functions' callback pattern (waitForTaskToken),
allowing long-running Bedrock jobs to complete asynchronously while
Step Functions waits. The task token is stored in DynamoDB and used
later by EventBridge rules to signal job completion.
"""

import os
from typing import Any

import boto3
from botocore.config import Config

import utils
from custom_types import JobConfig, TaskItem

# Configure structured logging for CloudWatch monitoring
logger = utils.get_logger()

# Configure aggressive retry strategy for Bedrock API calls
# Batch job creation can experience throttling when multiple jobs
# are submitted simultaneously by the Step Functions Map state
config = Config(
    retries={
        "max_attempts": 100,  # High retry count for resilience
        "mode": "standard"    # Exponential backoff with jitter
    }
)

# Initialize AWS service clients
bedrock_client = boto3.client("bedrock", config=config)
dynamodb = boto3.resource("dynamodb")

# DynamoDB table for tracking job state and task tokens
# This table enables asynchronous job completion callbacks
task_table = dynamodb.Table(os.environ["TASK_TABLE"])


def lambda_handler(event, context) -> dict[str, Any]:
    """Submit a Bedrock batch inference job and track it for async completion.

    This function is invoked by Step Functions using the waitForTaskToken
    integration pattern. It submits a batch job to Bedrock, stores tracking
    information in DynamoDB, and returns immediately. Later, when the job
    completes, EventBridge rules will use the stored task token to signal
    Step Functions to continue.

    Args:
        event: Step Functions event containing:
            - taskToken: Callback token for async completion signaling
            - taskInput: JobConfig with job parameters including:
                - job_name: Unique name for the Bedrock job
                - model_id: Bedrock model to use
                - s3_uri_input: Input JSONL file location
                - s3_uri_output: Output directory for results
                - input_parquet_path: Original data for postprocessing
        context: Lambda runtime context (unused but required by Lambda).

    Returns:
        Dictionary representation of TaskItem containing:
            - job_arn: ARN of the submitted Bedrock job
            - status: Initial job status (InProgress, Submitted, etc.)
            - All input parameters for downstream processing

    Note:
        The function will retry up to 100 times if throttled by Bedrock,
        ensuring reliable job submission even under high concurrency.
    """
    # Extract task token for async callback
    task_token = event["taskToken"]
    logger.info(f"Got task token {task_token}")
    # Validate and parse job configuration from Step Functions
    payload = JobConfig.model_validate(event["taskInput"])

    # Build additional job parameters based on environment configuration
    # Reference: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/create_model_invocation_job.html
    additional_kwargs = {}

    # Apply job timeout if configured (helps control costs and prevent stuck jobs)
    job_timeout_hours = int(os.environ.get("JOB_TIMEOUT_HOURS", -1))
    if job_timeout_hours > 0:
        additional_kwargs["timeoutDurationInHours"] = job_timeout_hours
        logger.info(f"Setting job timeout to {job_timeout_hours} hours")

    # Submit batch inference job to Bedrock
    # This is an asynchronous operation - the job runs in the background
    job_arn = bedrock_client.create_model_invocation_job(
        jobName=payload.job_name,                        # Unique job identifier
        roleArn=os.environ["BEDROCK_ROLE_ARN"],         # IAM role for Bedrock to access S3
        modelId=payload.model_id,                        # Model to use for inference
        inputDataConfig={
            "s3InputDataConfig": {
                "s3InputFormat": "JSONL",                # Bedrock expects JSONL format
                "s3Uri": payload.s3_uri_input,           # Location of preprocessed input
            }
        },
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": payload.s3_uri_output,          # Directory for job outputs
            }
        },
        **additional_kwargs,                             # Timeout and other optional params
    )["jobArn"]
    logger.info(f"Started job: {job_arn}")

    # Verify job submission and get initial status
    # This ensures the job was accepted by Bedrock before proceeding
    job_details = bedrock_client.get_model_invocation_job(
        jobIdentifier=job_arn,
    )
    logger.info(f"Job status: {job_details['status']}")

    # Create task tracking record with all necessary metadata
    # This record enables async job completion callbacks and postprocessing
    task_item = TaskItem(
        job_arn=job_arn,                               # Unique job identifier for status queries
        model_id=payload.model_id,                     # Model used (needed for output parsing)
        input_parquet_path=payload.input_parquet_path, # Original data for joining with outputs
        s3_uri_output=payload.s3_uri_output,           # Output location for postprocessing
        status=job_details["status"],                  # Initial status (InProgress, Submitted)
        error_message=None,                            # Will be populated if job fails
        task_token=task_token,                         # Step Functions callback token
    )

    # Store task in DynamoDB for EventBridge rules to retrieve
    # The job_arn is the partition key for efficient lookups
    logger.info("Updating task table")
    task_table.put_item(Item=task_item.model_dump(exclude_none=True))

    # Return task details for logging and debugging
    # Step Functions will wait for async callback via task token
    return task_item.model_dump(exclude_none=True)
