"""Azure integration: Blob Storage for document storage, Azure OpenAI for
managed models.

Persona: GenAI Developer.
Requires `azure-storage-blob` and Azure credentials. Not exercised
against a live Azure subscription in this workspace.
"""
from __future__ import annotations

import os

from src.cloud.base import CloudStorageProvider


class AzureBlobStorageProvider(CloudStorageProvider):
    name = "azure-blob"

    def __init__(self, container: str | None = None, connection_string: str | None = None):
        self.container = container or os.getenv("AZURE_STORAGE_CONTAINER", "documind-ai-raw-docs")
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from azure.storage.blob import ContainerClient
            except ImportError as exc:
                raise RuntimeError("azure-storage-blob not installed. `pip install azure-storage-blob`.") from exc
            if not self.connection_string:
                raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set.")
            self._client = ContainerClient.from_connection_string(self.connection_string, self.container)
        return self._client

    def upload_file(self, local_path: str, remote_key: str) -> str:
        with open(local_path, "rb") as fh:
            self.client.upload_blob(name=remote_key, data=fh, overwrite=True)
        return f"azure://{self.container}/{remote_key}"

    def download_file(self, remote_key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        blob_client = self.client.get_blob_client(remote_key)
        with open(local_path, "wb") as fh:
            fh.write(blob_client.download_blob().readall())
        return local_path

    def list_objects(self, prefix: str = "") -> list[str]:
        return [b.name for b in self.client.list_blobs(name_starts_with=prefix)]
