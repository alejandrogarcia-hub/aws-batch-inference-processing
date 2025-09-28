#!/usr/bin/env node

/**
 * AWS CDK Application Entry Point for Bedrock Batch Inference Processing
 *
 * This is the main entry point for the CDK application that deploys the infrastructure
 * required for orchestrating Amazon Bedrock batch inference jobs. The application creates
 * a complete serverless architecture including Step Functions, Lambda functions, DynamoDB
 * tables, and S3 buckets to handle large-scale batch processing of AI/ML inference requests.
 *
 * @module bedrock-batch-inference-processing
 * @requires source-map-support/register - Provides source map support for stack traces
 * @requires aws-cdk-lib - AWS CDK core library for infrastructure definition
 * @requires ../lib/bedrock-batch-inference-stack - Main stack definition
 *
 * @example
 * // Deploy with default settings from cdk.json:
 * npm run cdk -- deploy
 *
 * @example
 * // Deploy with custom context values:
 * npm run cdk -- deploy \
 *   --context maxSubmittedAndInProgressJobs=5 \
 *   --context bedrockBatchInferenceTimeoutHours=24
 *
 * @see {@link https://docs.aws.amazon.com/cdk/v2/guide/apps.html} - CDK Applications documentation
 * @see {@link https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html} - Bedrock Batch Inference
 */

// Enable source map support for better error stack traces in transpiled TypeScript code
import 'source-map-support/register';

import * as cdk from 'aws-cdk-lib';
import { BedrockBatchInferenceStack } from '../lib/bedrock-batch-inference-stack';

/**
 * Initialize the CDK application instance.
 * This creates the root construct that will contain all stacks.
 */
const app = new cdk.App();

/**
 * Retrieve the maximum number of concurrent batch inference jobs from CDK context.
 * This value controls the parallelism in the Step Functions Map state and prevents
 * overwhelming the Bedrock service with too many concurrent job submissions.
 *
 * @remarks
 * This is a required parameter that must be defined in cdk.json or passed via --context.
 * Typical values range from 1-10 depending on your AWS account limits.
 *
 * @throws {Error} Throws an error if the context variable is not defined
 */
const maxSubmittedAndInProgressJobs = app.node.tryGetContext('maxSubmittedAndInProgressJobs');
if (maxSubmittedAndInProgressJobs === undefined) {
  throw new Error('Missing required context variable: maxSubmittedAndInProgressJobs');
}

/**
 * Retrieve the memory allocation in MB for the preprocess lambda function.
 */
const preprocessFunctionMemoryMb = app.node.tryGetContext('preprocessFunctionMemoryMb');
if (preprocessFunctionMemoryMb === undefined) {
  throw new Error('Missing required context variable: preprocessFunctionMemoryMb');
}

/**
 * Retrieve the memory allocation in MB for the postprocess lambda function.
 */
const postprocessFunctionMemoryMb = app.node.tryGetContext('postprocessFunctionMemoryMb');
if (postprocessFunctionMemoryMb === undefined) {
  throw new Error('Missing required context variable: postprocessFunctionMemoryMb');
}

/**
 * Retrieve the optional timeout duration for Bedrock batch inference jobs.
 * If specified, this sets a maximum duration (in hours) for batch jobs to complete
 * before they are automatically terminated.
 *
 * @remarks
 * This is an optional parameter. If not specified, Bedrock will use its default
 * timeout behavior (typically 72 hours). Setting this value can help control costs
 * and ensure jobs don't run indefinitely.
 */
const bedrockBatchInferenceTimeoutHours = app.node.tryGetContext('bedrockBatchInferenceTimeoutHours');

/**
 * Instantiate the main CDK stack for the Bedrock Batch Inference Processing application.
 *
 * @remarks
 * The stack name 'BedrockBatchInferenceProcessingStack' will be used as the CloudFormation
 * stack name in AWS. This stack contains all the resources needed for the batch processing
 * workflow including:
 * - S3 buckets for input/output storage
 * - Lambda functions for preprocessing, job management, and postprocessing
 * - Step Functions state machine for orchestration
 * - DynamoDB table for job tracking
 * - EventBridge rules for job status monitoring
 * - IAM roles and policies for service permissions
 *
 * The stack accepts configuration through props that control the behavior of the
 * batch processing system.
 */
new BedrockBatchInferenceStack(app, 'BedrockBatchInferenceProcessingStack', {
  /**
   * Maximum number of Bedrock batch inference jobs that can be submitted or in progress
   * simultaneously. This prevents quota exhaustion and manages concurrency.
   */
  maxSubmittedAndInProgressJobs: Number(maxSubmittedAndInProgressJobs),

  /**
   * Memory allocation in MB for the preprocess lambda function.
   */
  preprocessFunctionMemoryMb: Number(preprocessFunctionMemoryMb),

  /**
   * Memory allocation in MB for the postprocess lambda function.
   */
  postprocessFunctionMemoryMb: Number(postprocessFunctionMemoryMb),

  /**
   * Optional timeout for batch inference jobs in hours.
   * If undefined, Bedrock's default timeout will be used.
   */
  bedrockBatchInferenceTimeoutHours: bedrockBatchInferenceTimeoutHours !== undefined
    ? Number(bedrockBatchInferenceTimeoutHours)
    : undefined,
});
