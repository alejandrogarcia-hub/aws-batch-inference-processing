/**
 * @fileoverview AWS CDK Stack for Bedrock Batch Inference Processing Infrastructure
 *
 * This module defines the complete infrastructure for orchestrating Amazon Bedrock batch
 * inference jobs at scale. It creates a serverless architecture that can process millions
 * of AI/ML inference requests cost-effectively using Bedrock's batch processing capabilities.
 *
 * @module bedrock-batch-inference-stack
 * @see {@link https://docs.aws.amazon.com/cdk/v2/guide/stacks.html} - CDK Stacks documentation
 * @see {@link https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html} - Bedrock Batch Inference
 */

import * as path from 'path';

import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as assets from "aws-cdk-lib/aws-ecr-assets";
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';


/**
 * Configuration properties for the Bedrock Batch Inference Stack.
 *
 * @interface BedrockBatchInferenceStackProps
 * @extends {cdk.StackProps}
 *
 * @example
 * ```typescript
 * new BedrockBatchInferenceStack(app, 'MyStack', {
 *   maxSubmittedAndInProgressJobs: 10,
 *   bedrockBatchInferenceTimeoutHours: 24
 * });
 * ```
 */
export interface BedrockBatchInferenceStackProps extends cdk.StackProps {
  /**
   * Maximum number of concurrent Bedrock batch inference jobs.
   *
   * Controls the parallelism in the Step Functions Map state and helps prevent
   * overwhelming the Bedrock service or hitting AWS account quotas.
   *
   * @remarks
   * Typical values range from 1-20 depending on your account limits and expected workload.
   * Higher values process jobs faster but may hit service quotas.
   */
  maxSubmittedAndInProgressJobs: number;

  /**
   * Memory allocation in MB for the preprocess function.
   */
  preprocessFunctionMemoryMb: number;

  /**
   * Memory allocation in MB for the postprocess function.
   */
  postprocessFunctionMemoryMb: number;

  /**
   * Optional timeout duration in hours for individual batch inference jobs.
   *
   * If specified, jobs will be automatically terminated if they exceed this duration.
   * If not specified, Bedrock's default timeout (typically 72 hours) will be used.
   *
   * @remarks
   * Setting this helps control costs and prevents runaway jobs.
   * Consider your largest expected batch size when setting this value.
   */
  bedrockBatchInferenceTimeoutHours?: number;
}


/**
 * Main CDK Stack class that creates all AWS resources required for Bedrock batch inference processing.
 *
 * This stack implements a complete serverless architecture for orchestrating large-scale batch
 * inference jobs using Amazon Bedrock. The architecture consists of:
 *
 * - **Storage Layer**: S3 buckets for input/output data and access logs
 * - **Compute Layer**: Lambda functions for preprocessing, job management, and postprocessing
 * - **Orchestration Layer**: Step Functions state machine for workflow coordination
 * - **Data Layer**: DynamoDB table for job state tracking
 * - **Event Layer**: EventBridge rules for asynchronous job status monitoring
 * - **Security Layer**: IAM roles and policies with least-privilege access
 * - **Monitoring Layer**: CloudWatch alarms for failure detection
 *
 * @class BedrockBatchInferenceStack
 * @extends {cdk.Stack}
 *
 * @remarks
 * The stack is designed to handle millions of inference requests efficiently by:
 * - Batching requests into JSONL files for cost-effective processing (50% discount)
 * - Managing concurrency to respect AWS service quotas
 * - Providing automatic retry and error handling
 * - Tracking job status asynchronously through EventBridge
 * - Supporting multiple model types (text generation and embeddings)
 */
export class BedrockBatchInferenceStack extends cdk.Stack {
  /**
   * Constructs a new instance of the Bedrock Batch Inference Stack.
   *
   * @param {Construct} scope - The parent construct (usually an App or Stage)
   * @param {string} id - The unique identifier for this stack
   * @param {BedrockBatchInferenceStackProps} props - Configuration properties for the stack
   *
   * @throws {Error} Throws if required context variables are missing or invalid
   */
  constructor(scope: Construct, id: string, props: BedrockBatchInferenceStackProps) {
    super(scope, id, props);

    /**
     * Path to Lambda function source code directory.
     * Contains Python code for all Lambda functions in the stack.
     */
    const lambdaAssetPath = path.join(__dirname, '../lambda');

    /**
     * Configure Lambda memory allocations based on regional quotas.
     *
     * Some AWS regions (notably eu-central-2) enforce a 3 GB memory limit for Lambda
     * functions on new accounts. These values can be overridden via CDK context if
     * your account has higher quotas.
     *
     * @remarks
     * Memory allocation affects:
     * - Processing speed (more memory = faster CPU)
     * - Maximum file sizes that can be processed
     * - Cost (billed per GB-second)
     */
    const preprocessMemoryMb = Number(props.preprocessFunctionMemoryMb ?? 3008);
    const postprocessMemoryMb = Number(props.postprocessFunctionMemoryMb ?? 3008);

    /**
     * Helper function to create Docker-based Lambda functions with consistent configuration.
     *
     * All Lambda functions in this stack use Docker containers to ensure consistent
     * Python runtime environments and dependency management.
     *
     * @param {string} lambdaId - Unique identifier for the Lambda function construct
     * @param {object} options - Configuration options for the Lambda function
     * @param {string} options.description - Human-readable description of the function's purpose
     * @param {string[]} options.cmd - Command array specifying the handler (e.g., ['module.handler'])
     * @param {Record<string, string>} [options.environment] - Environment variables for the function
     * @param {cdk.Duration} [options.timeout=5 minutes] - Maximum execution time for the function
     * @param {number} [options.memorySize] - Memory allocation in MB
     * @param {cdk.Size} [options.ephemeralStorageSize] - Temporary storage allocation (/tmp)
     *
     * @returns {lambda.DockerImageFunction} Configured Docker-based Lambda function
     *
     * @remarks
     * Uses Linux AMD64 platform for compatibility with Lambda runtime
     * Default timeout is 5 minutes if not specified
     */
    const createDockerLambda = (
      lambdaId: string,
      options: {
        description: string;
        cmd: string[];
        environment?: Record<string, string>;
        timeout?: cdk.Duration;
        memorySize?: number;
        ephemeralStorageSize?: cdk.Size;
      },
    ): lambda.DockerImageFunction => {
      return new lambda.DockerImageFunction(this, lambdaId, {
        description: options.description,
        code: lambda.DockerImageCode.fromImageAsset(lambdaAssetPath, {
          platform: assets.Platform.LINUX_AMD64,
          cmd: options.cmd,
        }),
        environment: options.environment,
        timeout: options.timeout ?? cdk.Duration.minutes(5),
        memorySize: options.memorySize,
        ephemeralStorageSize: options.ephemeralStorageSize,
      });
    };

    /**
     * S3 bucket for server access logs.
     *
     * This bucket stores access logs from the main data bucket, providing an audit trail
     * of all object operations. Required for compliance and security monitoring.
     *
     * @remarks
     * - Uses S3-managed encryption (SSE-S3)
     * - Enforces SSL/TLS for all requests
     * - Blocks all public access
     * - Enables versioning for log integrity
     * - Auto-deletes on stack removal (for non-production use)
     */
    const accessLogsBucket = new s3.Bucket(this, 'accessLogsBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    accessLogsBucket.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AWSLogDeliveryWrite',
      principals: [new iam.ServicePrincipal('delivery.logs.amazonaws.com')],
      actions: ['s3:PutObject'],
      resources: [accessLogsBucket.arnForObjects('batch-inference/*')],
      conditions: {
        StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' },
      },
    }));
    accessLogsBucket.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AWSLogDeliveryCheck',
      principals: [new iam.ServicePrincipal('delivery.logs.amazonaws.com')],
      actions: ['s3:GetBucketAcl', 's3:ListBucket'],
      resources: [accessLogsBucket.bucketArn],
    }));

    /**
     * Main S3 bucket for storing batch inference data.
     *
     * This bucket serves as the central storage location for:
     * - Input JSONL files for batch inference jobs
     * - Output results from completed inference jobs
     * - Intermediate Parquet files for efficient data processing
     * - Uploaded CSV/Parquet datasets from users
     *
     * @remarks
     * Directory structure:
     * - `/batch_inputs_json/` - JSONL files ready for Bedrock processing
     * - `/batch_outputs_json/` - Raw outputs from Bedrock jobs
     * - `/batch_inputs_parquet/` - Parquet format inputs for joining
     * - `/batch_output_parquet/` - Final processed results in Parquet format
     * - `/inputs/` - User-uploaded source datasets
     */
    const bucket = new s3.Bucket(this, 'bucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      serverAccessLogsBucket: accessLogsBucket,
      serverAccessLogsPrefix: 'batch-inference/',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    /**
     * Helper function to grant granular S3 bucket access to IAM principals.
     *
     * Implements least-privilege access by granting permissions only to specific
     * object prefixes rather than the entire bucket.
     *
     * @param {iam.IGrantable} grantee - IAM principal to grant access to
     * @param {string[]} prefixes - S3 object prefixes to allow access to
     * @param {string[]} objectActions - S3 actions to allow (e.g., 's3:GetObject')
     * @param {boolean} [includeDelete=false] - Whether to include delete permissions
     *
     * @remarks
     * This function adds both object-level and bucket-level (ListBucket) permissions
     * with appropriate conditions to restrict access to specified prefixes only.
     */
    const grantBucketAccess = (
      grantee: iam.IGrantable,
      prefixes: string[],
      objectActions: string[],
      includeDelete = false,
    ): void => {
      const sanitizedPrefixes = Array.from(new Set(prefixes));
      if (sanitizedPrefixes.length === 0) {
        return;
      }

      const objectResources = sanitizedPrefixes.map(prefix => bucket.arnForObjects(prefix));
      const actions = includeDelete ? [...objectActions, 's3:DeleteObject'] : objectActions;
      grantee.grantPrincipal.addToPrincipalPolicy(new iam.PolicyStatement({
        actions,
        resources: objectResources,
      }));
      grantee.grantPrincipal.addToPrincipalPolicy(new iam.PolicyStatement({
        actions: ['s3:ListBucket'],
        resources: [bucket.bucketArn],
        conditions: {
          StringLike: { 's3:prefix': sanitizedPrefixes },
        },
      }));
    };

    /**
     * DynamoDB table for tracking job state and task tokens.
     *
     * Maps Bedrock job ARNs to Step Functions task tokens, enabling asynchronous
     * communication between EventBridge job status updates and the waiting Step Function tasks.
     *
     * @remarks
     * - Partition key: job_arn (string) - Unique identifier for each Bedrock job
     * - Stores task tokens for callback pattern implementation
     * - Auto-deletes on stack removal (for non-production use)
     */
    const taskTable = new dynamodb.TableV2(this, 'taskTable', {
      partitionKey: { name: 'job_arn', type: dynamodb.AttributeType.STRING },
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    /**
     * IAM service role for Bedrock batch inference jobs.
     *
     * This role is assumed by the Bedrock service when executing batch inference jobs.
     * It grants necessary permissions for Bedrock to read input files and write outputs.
     *
     * @remarks
     * Permissions include:
     * - Read access to input JSONL files
     * - Write access for output results
     * - Cross-region model invocation for inference profiles
     */
    const bedrockServiceRole = new iam.Role(this, 'bedrockServiceRole', {
      assumedBy: new iam.ServicePrincipal('bedrock.amazonaws.com'),
    });
    grantBucketAccess(
      bedrockServiceRole,
      ['batch_inputs_json/*', 'batch_outputs_json/*'],
      ['s3:GetObject', 's3:PutObject'],
      true,
    );

    /**
     * Configure model invocation permissions for cross-region inference.
     *
     * These permissions allow the Bedrock service to invoke specific foundation models
     * and inference profiles, including cross-region profiles (e.g., us.* profiles).
     *
     * @see {@link https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html} - Bedrock IAM documentation
     *
     * @remarks
     * Add additional model ARNs here as needed for your use case.
     * Current models:
     * - Claude 3 Haiku (base model)
     * - Claude 3.5 Haiku (cross-region inference profile)
     * - Titan Embeddings V2
     */
    bedrockServiceRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['bedrock:InvokeModel'],
      resources: [
        'arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0',
        'arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0',
        'arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0',
      ],
    }));

    // lambda functions
    const preprocessFunction = createDockerLambda('preprocessFunction', {
      description: 'Prepare the bedrock batch input files',
      cmd: ['preprocess.lambda_handler'],
      environment: {
        BUCKET_NAME: bucket.bucketName,
        HF_HOME: '/tmp/huggingface'
      },
      timeout: cdk.Duration.minutes(15),
      memorySize: preprocessMemoryMb,
      ephemeralStorageSize: cdk.Size.mebibytes(512),
    });
    grantBucketAccess(
      preprocessFunction,
      [
        'hf/*',
        'batch_inputs_json/*',
        'batch_inputs_parquet/*',
        'batch_outputs_json/*',
        'batch_output_parquet/*',
        'inputs/*',
      ],
      ['s3:GetObject', 's3:PutObject'],
      true,
    );

    const startBatchInferenceFunction = createDockerLambda('startBatchInferenceFunction', {
      description: 'Starts the bedrock batch inference jobs',
      cmd: ['start_batch_inference_job.lambda_handler'],
      environment: {
        BEDROCK_ROLE_ARN: bedrockServiceRole.roleArn,
        TASK_TABLE: taskTable.tableName,
        JOB_TIMEOUT_HOURS: (props.bedrockBatchInferenceTimeoutHours ?? -1).toString(),
      },
      timeout: cdk.Duration.minutes(5),
    });
    taskTable.grantReadWriteData(startBatchInferenceFunction);

    const modelResourceArns = [
      'arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0',
      'arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0',
      'arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0',
    ];
    const bedrockJobArn = cdk.Stack.of(this).formatArn({
      service: 'bedrock',
      resource: 'model-invocation-job',
      resourceName: '*',
    });

    startBatchInferenceFunction.addToRolePolicy(new iam.PolicyStatement({
      actions: ['bedrock:CreateModelInvocationJob'],
      effect: iam.Effect.ALLOW,
      resources: [...modelResourceArns, bedrockJobArn],
    }));
    startBatchInferenceFunction.addToRolePolicy(new iam.PolicyStatement({
      actions: ['bedrock:GetModelInvocationJob'],
      effect: iam.Effect.ALLOW,
      resources: [bedrockJobArn],
    }));
    startBatchInferenceFunction.addToRolePolicy(new iam.PolicyStatement({
      actions: ['iam:PassRole'],
      effect: iam.Effect.ALLOW,
      resources: [bedrockServiceRole.roleArn], // Reference to your service role
    }));

    // event source for completed jobs
    const batchJobCompleteRule = new events.Rule(this, 'batchJobCompleteRule', {
      eventPattern: {
        source: ['aws.bedrock'],
        detailType: ['Batch Inference Job State Change'],
      },
    });

    const getBatchInferenceFunction = createDockerLambda('getBatchInferenceFunction', {
      description: 'Monitors the progress of bedrock batch inference jobs',
      cmd: ['get_batch_inference_job.lambda_handler'],
      timeout: cdk.Duration.seconds(15),
      environment: {
        TASK_TABLE: taskTable.tableName,
      }
    });
    batchJobCompleteRule.addTarget(new targets.LambdaFunction(getBatchInferenceFunction));
    taskTable.grantReadWriteData(getBatchInferenceFunction);
    getBatchInferenceFunction.addToRolePolicy(new iam.PolicyStatement({
      actions: ['bedrock:GetModelInvocationJob'],
      effect: iam.Effect.ALLOW,
      resources: [bedrockJobArn],
    }));

    const postprocessFunction = createDockerLambda('postprocessFunction', {
      description: 'Process the bedrock batch output files',
      cmd: ['postprocess.lambda_handler'],
      memorySize: postprocessMemoryMb,
      environment: {
        BUCKET_NAME: bucket.bucketName,
      },
      timeout: cdk.Duration.minutes(5),
    });
    grantBucketAccess(
      postprocessFunction,
      ['batch_inputs_parquet/*', 'batch_outputs_json/*', 'batch_output_parquet/*'],
      ['s3:GetObject', 's3:PutObject'],
      true,
    );

    // ENTRY: step function tasks
    // follow tasks to reach the next phase
    const preprocessTask = new tasks.LambdaInvoke(this, 'preprocessTask', {
      lambdaFunction: preprocessFunction,
      outputPath: '$.Payload',
    });

    const startBatchInferenceTask = new tasks.LambdaInvoke(this, 'startBatchInferenceTask', {
      lambdaFunction: startBatchInferenceFunction,
      integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
      payload: sfn.TaskInput.fromObject({
        taskToken: sfn.JsonPath.taskToken,
        taskInput: sfn.JsonPath.stringAt('$'),
      }),
    });

    // explicit retries to handle throttling errors in particular
    startBatchInferenceTask.addRetry({
      maxAttempts: 3,
    });

    const postprocessMap = new sfn.Map(this, 'postprocessMap', {
      maxConcurrency: props.maxSubmittedAndInProgressJobs,
      itemsPath: sfn.JsonPath.stringAt('$.completed_jobs'),
      resultPath: '$.output_paths',
    });

    const postprocessTask = new tasks.LambdaInvoke(this, 'postprocessTask', {
      lambdaFunction: postprocessFunction,
      outputPath: '$.Payload',
    }).addRetry({
      maxAttempts: 3,
      backoffRate: 2,
      interval: cdk.Duration.seconds(5),
    });

    // step function
    const batchProcessingMap = new sfn.Map(this, 'batchProcessingMap', {
      maxConcurrency: props.maxSubmittedAndInProgressJobs,
      itemsPath: sfn.JsonPath.stringAt('$.jobs'),
      resultPath: '$.completed_jobs',
    });

    const chain = preprocessTask
      .next(batchProcessingMap.itemProcessor(startBatchInferenceTask))
      .next(postprocessMap.itemProcessor(postprocessTask));

    // state machine
    const stepFunction = new sfn.StateMachine(this, 'bedrockBatchInferenceProcessingSfn', {
      definitionBody: sfn.DefinitionBody.fromChainable(chain),
    });

    stepFunction.grantTaskResponse(getBatchInferenceFunction);

    // CloudWatch alarm to highlight failed executions for visibility.
    new cloudwatch.Alarm(this, 'stateMachineFailuresAlarm', {
      metric: stepFunction.metricFailed({ period: cdk.Duration.minutes(5) }),
      threshold: 1,
      evaluationPeriods: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
      alarmDescription: 'Triggers when the Bedrock batch inference processing workflow reports a failed execution.',
    });

    // output the state machine name & bucket name
    new cdk.CfnOutput(this, 'stepFunctionName', {
      value: stepFunction.stateMachineName,
    });
    new cdk.CfnOutput(this, 'bucketName', {
      value: bucket.bucketName,
    });
  }
}
