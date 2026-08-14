"""Cloud provider abstraction shared by AWS/Azure/GCP integration modules.

Persona: GenAI Developer / Solution Architect.
JD requirement covered: "Collaborate with cloud platforms (e.g., AWS,
Azure, GCP) to build Generative AI (GenAI) applications."

Every concrete provider implements the same `CloudStorageProvider`
interface for raw-document storage, so application code (the ETL
pipeline, the API layer) never imports `boto3`/`azure-storage-blob`/
`google-cloud-storage` directly — only `src/cloud/*` does. This is what
makes NFR-6 ("Portability") and risk R-3 ("vendor lock-in") mitigations
real rather than aspirational.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class CloudStorageProvider(ABC):
    """Object storage operations needed by the ETL pipeline (raw docs +
    manifests). Concrete providers: S3 (AWS), Blob Storage (Azure), GCS (GCP)."""

    @abstractmethod
    def upload_file(self, local_path: str, remote_key: str) -> str:
        """Upload a local file, return the remote URI."""

    @abstractmethod
    def download_file(self, remote_key: str, local_path: str) -> str:
        """Download a remote object to a local path, return the local path."""

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[str]:
        ...


class LocalFilesystemProvider(CloudStorageProvider):
    """Default provider for local dev/CI — mirrors the same interface so
    the pipeline can run without any cloud account configured, and the
    exact same call sites work unchanged once a real cloud provider is
    selected via `CLOUD_PROVIDER`."""

    name = "local"

    def __init__(self, root_dir: str = "data/cloud_sim"):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def upload_file(self, local_path: str, remote_key: str) -> str:
        import shutil

        dest = os.path.join(self.root_dir, remote_key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local_path, dest)
        return f"local://{dest}"

    def download_file(self, remote_key: str, local_path: str) -> str:
        import shutil

        src = os.path.join(self.root_dir, remote_key)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.copyfile(src, local_path)
        return local_path

    def list_objects(self, prefix: str = "") -> list[str]:
        results = []
        base = os.path.join(self.root_dir, prefix)
        if not os.path.isdir(base):
            return results
        for dirpath, _, filenames in os.walk(base):
            for f in filenames:
                results.append(os.path.relpath(os.path.join(dirpath, f), self.root_dir))
        return results


def get_cloud_storage_provider(name: str | None = None) -> CloudStorageProvider:
    provider = name or os.getenv("CLOUD_PROVIDER", "local")
    if provider == "local":
        return LocalFilesystemProvider()
    if provider == "aws":
        from src.cloud.aws_integration import S3StorageProvider

        return S3StorageProvider()
    if provider == "azure":
        from src.cloud.azure_integration import AzureBlobStorageProvider

        return AzureBlobStorageProvider()
    if provider == "gcp":
        from src.cloud.gcp_integration import GCSStorageProvider

        return GCSStorageProvider()
    raise ValueError(f"Unknown CLOUD_PROVIDER '{provider}'. Options: ['local', 'aws', 'azure', 'gcp']")
