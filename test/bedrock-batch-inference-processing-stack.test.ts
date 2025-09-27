import { App } from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';

import { BedrockBatchInferenceStack } from '../lib/bedrock-batch-inference-stack';

describe('BedrockBatchInferenceProcessingStack', () => {
  it('synthesizes the core resources', () => {
    const app = new App();
    const stack = new BedrockBatchInferenceStack(app, 'TestStack', {
      maxSubmittedAndInProgressJobs: 1,
    });

    const template = Template.fromStack(stack);

    template.resourceCountIs('AWS::S3::Bucket', 2);
    template.resourceCountIs('AWS::DynamoDB::GlobalTable', 1);
    template.resourceCountIs('AWS::StepFunctions::StateMachine', 1);

    template.hasResourceProperties('AWS::S3::Bucket', Match.objectLike({
      BucketEncryption: Match.objectLike({
        ServerSideEncryptionConfiguration: Match.arrayWith([
          Match.objectLike({
            ServerSideEncryptionByDefault: Match.objectLike({
              SSEAlgorithm: 'AES256',
            }),
          }),
        ]),
      }),
      LoggingConfiguration: Match.objectLike({
        DestinationBucketName: Match.anyValue(),
      }),
      VersioningConfiguration: Match.objectLike({
        Status: 'Enabled',
      }),
    }));
  });
});
