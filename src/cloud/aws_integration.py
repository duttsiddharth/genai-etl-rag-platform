"""AWS integration: S3 for document storage, Bedrock for managed models.

Persona: GenAI Developer.
Requires `boto3` and AWS credentials (via env vars, shared config, or an
IAM role when deployed to EKS — see `infra/terraform/`). Not exercised
against a live AWS account in this workspace; structurally complete and
unit-testable via mocking (`tests/unit/test_cloud_providers.py`).
"""
from __future__ import annotations

import os

from src.cloud.base import CloudStorageProvider


class S3StorageProvider(CloudStorageProvider):
    name = "aws-s3"

    def __init__(self, bucket: str | None = None, region: str | None = None):
        self.bucket = bucket or os.getenv("AWS_S3_BUCKET", "documind-ai-raw-docs")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("boto3 not installed. `pip install boto3`.") from exc
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def upload_file(self, local_path: str, remote_key: str) -> str:
        self.client.upload_file(local_path, self.bucket, remote_key)
        return f"s3://{self.bucket}/{remote_key}"

    def download_file(self, remote_key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        self.client.download_file(self.bucket, remote_key, local_path)
        return local_path

    def list_objects(self, prefix: str = "") -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys


def invoke_bedrock_model(prompt: str, model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0", region: str | None = None) -> str:
    """Convenience helper used by ops scripts / notebooks — the API layer
    goes through `src.genai.llm.BedrockClaudeProvider` instead."""
    import json

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 not installed. `pip install boto3`.") from exc

    client = boto3.client("bedrock-runtime", region_name=region or os.getenv("AWS_REGION", "us-east-1"))
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    response = client.invoke_model(modelId=model_id, body=body)
    return json.loads(response["body"].read())["content"][0]["text"]
