from src.cloud.base import CloudStorageProvider, LocalFilesystemProvider, get_cloud_storage_provider


def test_local_provider_implements_interface():
    provider = LocalFilesystemProvider(root_dir="/tmp/documind_test_cloud")
    assert isinstance(provider, CloudStorageProvider)


def test_local_provider_upload_download_roundtrip(tmp_path):
    provider = LocalFilesystemProvider(root_dir=str(tmp_path / "cloud_sim"))
    src_file = tmp_path / "source.txt"
    src_file.write_text("hello cloud")

    uri = provider.upload_file(str(src_file), "docs/source.txt")
    assert uri.startswith("local://")

    dest_file = tmp_path / "downloaded.txt"
    provider.download_file("docs/source.txt", str(dest_file))
    assert dest_file.read_text() == "hello cloud"

    assert "docs/source.txt" in provider.list_objects("docs")


def test_get_cloud_storage_provider_defaults_to_local():
    provider = get_cloud_storage_provider()
    assert provider.name == "local"


def test_get_cloud_storage_provider_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        get_cloud_storage_provider("not-a-cloud")


def test_all_cloud_providers_share_interface():
    """Structural check that AWS/Azure/GCP providers satisfy the same
    interface without requiring their optional SDKs to be installed."""
    from src.cloud.aws_integration import S3StorageProvider
    from src.cloud.azure_integration import AzureBlobStorageProvider
    from src.cloud.gcp_integration import GCSStorageProvider

    for cls in (S3StorageProvider, AzureBlobStorageProvider, GCSStorageProvider):
        assert issubclass(cls, CloudStorageProvider)
        assert hasattr(cls, "upload_file")
        assert hasattr(cls, "download_file")
        assert hasattr(cls, "list_objects")
