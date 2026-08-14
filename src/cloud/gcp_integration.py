"""GCP integration: Cloud Storage for document storage, Vertex AI for
managed models.

Persona: GenAI Developer.
Requires `google-cloud-storage` and GCP credentials. Not exercised
against a live GCP project in this workspace.
"""
from __future__ import annotations

import os

from src.cloud.base import CloudStorageProvider


class GCSStorageProvider(CloudStorageProvider):
    name = "gcp-gcs"

    def __init__(self, bucket: str | None = None, project: str | None = None):
        self.bucket_name = bucket or os.getenv("GCS_BUCKET", "documind-ai-raw-docs")
        self.project = project or os.getenv("GCP_PROJECT_ID")
        self._client = None

    @property
    def bucket(self):
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError("google-cloud-storage not installed. `pip install google-cloud-storage`.") from exc
            self._client = storage.Client(project=self.project)
        return self._client.bucket(self.bucket_name)

    def upload_file(self, local_path: str, remote_key: str) -> str:
        blob = self.bucket.blob(remote_key)
        blob.upload_from_filename(local_path)
        return f"gs://{self.bucket_name}/{remote_key}"

    def download_file(self, remote_key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        blob = self.bucket.blob(remote_key)
        blob.download_to_filename(local_path)
        return local_path

    def list_objects(self, prefix: str = "") -> list[str]:
        return [b.name for b in self.bucket.list_blobs(prefix=prefix)]
