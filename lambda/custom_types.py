from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _BaseModel(BaseModel):
    """Shared model behaviour for custom types."""

    model_config = ConfigDict(
        extra="allow", populate_by_name=True, str_strip_whitespace=True
    )

    def __getitem__(
        self, item: str
    ) -> Any:  # pragma: no cover - convenience for legacy dict-style access
        return getattr(self, item)

    def __contains__(self, item: object) -> bool:  # pragma: no cover
        return hasattr(self, str(item))


class JobStatus(str, Enum):
    """Bedrock batch invocation job lifecycle states."""

    SUBMITTED = "Submitted"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    PARTIALLY_COMPLETED = "PartiallyCompleted"
    EXPIRED = "Expired"
    VALIDATING = "Validating"
    SCHEDULED = "Scheduled"


class JobInput(_BaseModel):
    """State machine input parameters consumed by the preprocess Lambda."""

    s3_uri: str | None = Field(
        default=None,
        description="S3 URI pointing to a CSV or Parquet object to process. Mutually exclusive with dataset_id.",
    )
    dataset_id: str | None = Field(
        default=None,
        description="Hugging Face dataset identifier to pull instead of using an S3 object.",
    )
    split: str | None = Field(
        default="train",
        description="Dataset split to load when dataset_id is supplied.",
    )
    job_name_prefix: str = Field(
        description="Prefix used to create unique Bedrock batch job names.",
    )
    model_id: str = Field(
        description="Foundation model ID (for example anthropic.claude-3-haiku-20240307-v1:0).",
    )
    prompt_id: str | None = Field(
        default=None,
        description="Identifier of the prompt template for text models. Omit or null for embedding jobs.",
    )
    max_num_jobs: int | None = Field(
        default=None,
        description="Upper bound on the number of batch jobs emitted during preprocessing.",
    )
    max_records_per_job: int | None = Field(
        default=None,
        description="Chunk size used when splitting the input dataset into batch jobs.",
    )
    max_records_total: int | None = Field(
        default=None,
        description="Maximum number of records to process across all jobs. None processes the entire dataset.",
    )

    @classmethod
    def safe_validate(cls, data: dict[str, Any]) -> JobInput:
        """Validate incoming payloads tolerating partial data (used in tests & helper calls)."""

        merged = {
            "job_name_prefix": data.get("job_name_prefix", ""),
            "model_id": data.get("model_id", ""),
            **data,
        }
        return cls.model_validate(merged)


class BatchInferenceRecord(_BaseModel):
    """Structure for a single JSONL record submitted to Bedrock batch inference."""

    recordId: str = Field(
        description="Unique identifier that ties the output back to the original input row."
    )
    modelInput: dict[str, Any] = Field(
        description="Model-specific input payload encoded as a JSON object."
    )


class JobConfig(_BaseModel):
    """Job configuration emitted by preprocessing and consumed by the start-job Lambda."""

    model_id: str = Field(description="Foundation model identifier to execute.")
    job_name: str = Field(description="Unique job name submitted to Bedrock.")
    input_parquet_path: str = Field(
        description="S3 URI of the Parquet chunk that will be post-processed later."
    )
    s3_uri_input: str = Field(
        description="S3 URI pointing to the JSONL input for the batch job."
    )
    s3_uri_output: str = Field(
        description="Destination S3 prefix where Bedrock will write job results."
    )


class JobConfigList(_BaseModel):
    """Wrapper collection of prepared Bedrock batch job configurations."""

    jobs: list[JobConfig] = Field(
        default_factory=list, description="List of individual job configurations."
    )


class TaskItem(_BaseModel):
    """Metadata tracked for each submitted Bedrock batch inference job."""

    job_arn: str = Field(description="ARN of the Bedrock model invocation job.")
    model_id: str = Field(description="Model ID associated with the job.")
    input_parquet_path: str = Field(
        description="S3 URI to the Parquet chunk produced during preprocessing."
    )
    s3_uri_output: str = Field(
        description="S3 destination prefix that will contain Bedrock outputs."
    )
    status: JobStatus | None = Field(
        default=None,
        description="Latest known Bedrock job status (Completed, Failed, InProgress, etc.).",
    )
    error_message: str | None = Field(
        default=None,
        description="Error string recorded when the job transitions into a terminal failure state.",
    )
    task_token: str = Field(
        description="Step Functions task token tied to the job for callbacks."
    )


class CompletedJobsList(_BaseModel):
    """Collection of jobs handed off to the post-processing Lambda."""

    completed_jobs: list[TaskItem] = Field(
        default_factory=list,
        description="Completed job metadata objects to post-process.",
    )
