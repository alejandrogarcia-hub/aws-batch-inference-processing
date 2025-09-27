#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { BedrockBatchInferenceStack } from '../lib/bedrock-batch-inference-stack';

const app = new cdk.App();

const maxSubmittedAndInProgressJobs = app.node.tryGetContext('maxSubmittedAndInProgressJobs');
if (maxSubmittedAndInProgressJobs === undefined) {
  throw new Error('Missing required context variable: maxSubmittedAndInProgressJobs');
}

const bedrockBatchInferenceTimeoutHours = app.node.tryGetContext('bedrockBatchInferenceTimeoutHours');

new BedrockBatchInferenceStack(app, 'BedrockBatchInferenceProcessingStack', {
  maxSubmittedAndInProgressJobs: Number(maxSubmittedAndInProgressJobs),
  bedrockBatchInferenceTimeoutHours: bedrockBatchInferenceTimeoutHours !== undefined
    ? Number(bedrockBatchInferenceTimeoutHours)
    : undefined,
});
