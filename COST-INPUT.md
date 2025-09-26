# AWS Pricing Calculator Inputs – Europe (Zurich)

Use this sheet to reproduce the sample monthly estimate in the AWS Pricing Calculator. Adjust any values to match your actual workload.

| Service | Region | Key Inputs | Notes |
| --- | --- | --- | --- |
| **AWS Lambda** | Europe (Zurich) `eu-central-2` |<ul><li>Preprocess function: 10,240 MB memory, 300 s average duration, 30 invocations/mo.</li><li>Start job function: 512 MB memory, 3 s duration, 1,500 invocations/mo.</li><li>Get job status function: 512 MB memory, 3 s duration, 1,500 invocations/mo.</li><li>Postprocess function: 10,240 MB memory, 60 s duration, 1,500 invocations/mo.</li></ul> | Request count pricing is negligible at this scale. Adjust durations/memory if you right-size the functions. |
| **AWS Step Functions (Standard)** | Europe (Zurich) `eu-central-2` | 13,500 state transitions per month | Assumes ~450 transitions per daily run × 30 runs. |
| **Amazon S3 (Standard)** | Europe (Zurich) `eu-central-2` | 10 GB-month of storage; 200,000 PUT/COPY/POST/LIST; 200,000 GET | Represents holding 10 GB of batch input/output data for a full month. |
| **Amazon DynamoDB (On-demand)** | Europe (Zurich) `eu-central-2` | 15,000 write request units; 9,000 read request units | Captures per-job inserts/updates for the task tracking table. |
| **Amazon CloudWatch Logs** | Europe (Zurich) `eu-central-2` | 1 GB ingested; 1 GB archived | Rough allocation for Lambda/Step Functions logs. |
| **Amazon CloudWatch Alarms** | Europe (Zurich) `eu-central-2` | 1 standard alarm | For the Step Functions failure alarm. |
| **Amazon Bedrock – Titan Embeddings** | Europe (Zurich) `eu-central-2` | 384M tokens/mo (`50,000 records × 256 tokens × 30 runs`) | On-demand price used: $0.00011 per 1,000 tokens. Adjust for your token volume and chosen model. |

### How to Enter These Values

1. Open the [AWS Pricing Calculator](https://calculator.aws/).
2. Add each service listed above, choose the **Europe (Zurich)** region, and enter the usage metrics from the table.
3. For Lambda, create four function groups (preprocess, start-job, get-job, postprocess) and input memory, duration, and monthly invocations for each. The calculator will total the GB-seconds automatically.
4. For Step Functions, select *Standard workflow* and enter the monthly state transition count.
5. For S3, select *S3 Standard* storage, specify the average storage size, and add request counts (PUT/COPY/POST/LIST and GET). Leave data transfer at zero unless you expect cross-region traffic.
6. For DynamoDB, choose *On-demand (Pay-per-request)* capacity mode and enter the monthly read/write request totals.
7. For CloudWatch Logs, add ingestion and archival GB as needed (default retention incurs archive pricing after 30 days).
8. For CloudWatch Alarms, add the quantity of standard alarms you plan to keep enabled.

Adjust any metric to match your workload; the monthly cost scales roughly linearly with the number of batch runs and the average runtime of the Lambda functions. |
