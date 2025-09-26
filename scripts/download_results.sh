#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
Usage: scripts/download_results.sh <bucket-name> <job-prefix> [destination-dir]

Synchronizes result artifacts from the batch orchestration bucket to a local folder.

Examples:
  scripts/download_results.sh batch-inference-bucket-123456789012 my-job outputs/my-job
  scripts/download_results.sh batch-inference-bucket-123456789012 test-run

The script requires the AWS CLI and credentials with s3:GetObject permissions.
USAGE
  exit 1
fi

bucket_name="$1"
job_prefix="$2"
dest_dir="${3:-downloads/${job_prefix}}"

echo "Downloading results for job prefix '${job_prefix}' from bucket '${bucket_name}'"
echo "Destination directory: ${dest_dir}"

mkdir -p "${dest_dir}"

aws s3 sync "s3://${bucket_name}/batch_output_parquet/${job_prefix}" "${dest_dir}/parquet" --exact-timestamps
aws s3 sync "s3://${bucket_name}/batch_outputs_json/${job_prefix}" "${dest_dir}/json" --exact-timestamps

echo "Sync complete. Files stored under ${dest_dir}" 
