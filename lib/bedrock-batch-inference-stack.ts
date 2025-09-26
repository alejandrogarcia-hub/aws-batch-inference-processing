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


export interface BedrockBatchInferenceStackProps extends cdk.StackProps {
  maxSubmittedAndInProgressJobs: number;
  bedrockBatchInferenceTimeoutHours?: number;
}


export class BedrockBatchInferenceStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: BedrockBatchInferenceStackProps) {
    super(scope, id, props);

    const lambdaAssetPath = path.join(__dirname, '../lambda');

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

    // server access logs bucket for auditing object access
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
      principals: [new iam.ServicePrincipal('logdelivery.s3.amazonaws.com')],
      actions: ['s3:PutObject'],
      resources: [accessLogsBucket.arnForObjects('batch-inference/*')],
      conditions: {
        StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' },
      },
    }));
    accessLogsBucket.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AWSLogDeliveryCheck',
      principals: [new iam.ServicePrincipal('logdelivery.s3.amazonaws.com')],
      actions: ['s3:GetBucketAcl', 's3:ListBucket'],
      resources: [accessLogsBucket.bucketArn],
    }));

    // bucket storing batch job inputs/outputs
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

    // dynamo table for job arn -> task tokens
    const taskTable = new dynamodb.TableV2(this, 'taskTable', {
      partitionKey: { name: 'job_arn', type: dynamodb.AttributeType.STRING },
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // service role for bedrock batch inference
    const bedrockServiceRole = new iam.Role(this, 'bedrockServiceRole', {
      assumedBy: new iam.ServicePrincipal('bedrock.amazonaws.com'),
    });
    grantBucketAccess(
      bedrockServiceRole,
      ['batch_inputs_json/*', 'batch_outputs_json/*'],
      ['s3:GetObject', 's3:PutObject'],
      true,
    );

    // allow cross-region inference: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html#batch-iam-sr-identity
    // add permissions for additional models as needed
    bedrockServiceRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ['bedrock:InvokeModel'],
      resources: [
        'arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0',
        'arn:aws:bedrock:*::inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0',
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
      memorySize: 10240,  // recommend a large amount of memory if using max. batch sizes (50k records)
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
      'arn:aws:bedrock:*::inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0',
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
      resources: modelResourceArns,
    }));
    startBatchInferenceFunction.addToRolePolicy(new iam.PolicyStatement({
      actions: ['bedrock:GetModelInvocationJob'],
      effect: iam.Effect.ALLOW,
      resources: [bedrockJobArn],
    }));
    startBatchInferenceFunction.addToRolePolicy(new iam.PolicyStatement({
      actions: ['iam:PassRole'],
      resources: [bedrockServiceRole.roleArn], // Reference to your service role
      effect: iam.Effect.ALLOW,
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
      memorySize: 10240,
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

    // step function tasks
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
    const stepFunction = new sfn.StateMachine(this, 'bedrockBatchOrchestratorSfn', {
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
      alarmDescription: 'Triggers when the Bedrock batch orchestrator reports a failed execution.',
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
