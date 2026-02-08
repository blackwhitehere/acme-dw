"""Storage backends for acme-dw.

Provides a StorageBackend ABC and concrete implementations for local
filesystem and S3 storage.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import shutil


class StorageBackend(ABC):
    """Abstract base class for data warehouse storage backends."""

    @abstractmethod
    def upload_file(self, local_path: Union[str, Path], key: str, **kwargs) -> None:
        """Upload a local file to the storage backend."""

    @abstractmethod
    def upload_files(self, file_mappings: dict[str, str], **kwargs) -> None:
        """Upload multiple local files. file_mappings: {local_path: key}."""

    @abstractmethod
    def download_file(self, key: str, local_path: Union[str, Path], **kwargs) -> None:
        """Download a file from the storage backend to a local path."""

    @abstractmethod
    def download_files(self, file_mappings: dict[str, str], **kwargs) -> None:
        """Download multiple files. file_mappings: {key: local_path}."""

    @abstractmethod
    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys under a prefix."""

    @abstractmethod
    def path_exists(self, key: str) -> bool:
        """Check whether a key exists in the backend."""


class LocalBackend(StorageBackend):
    """Filesystem-based storage backend.

    Stores files under a root directory, using the key as a relative path.

    Args:
        root_dir: Root directory for all stored files.
    """

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        return self.root / key

    def upload_file(self, local_path: Union[str, Path], key: str, **kwargs) -> None:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dest))

    def upload_files(self, file_mappings: dict[str, str], **kwargs) -> None:
        for local_path, key in file_mappings.items():
            self.upload_file(local_path, key)

    def download_file(self, key: str, local_path: Union[str, Path], **kwargs) -> None:
        src = self._resolve(key)
        if not src.exists():
            raise FileNotFoundError(f"File not found in local backend: {key}")
        shutil.copy2(str(src), str(local_path))

    def download_files(self, file_mappings: dict[str, str], **kwargs) -> None:
        for key, local_path in file_mappings.items():
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.download_file(key, dest)

    def list_objects(self, prefix: str) -> list[str]:
        prefix_path = self._resolve(prefix)
        if not prefix_path.exists():
            return []
        results = []
        for p in prefix_path.rglob("*"):
            if p.is_file():
                results.append(str(p.relative_to(self.root)))
        return sorted(results)

    def path_exists(self, key: str) -> bool:
        return self._resolve(key).exists()


class S3Backend(StorageBackend):
    """S3-based storage backend wrapping acme_s3.S3Client.

    Args:
        bucket_name: S3 bucket name.
        s3_client_kwargs: Optional kwargs passed to S3Client constructor.
    """

    def __init__(self, bucket_name: str, s3_client_kwargs: dict | None = None) -> None:
        try:
            from acme_s3 import S3Client
        except ImportError:
            raise ImportError(
                "acme_s3 is required for S3 storage. "
                "Install it with: pip install acme_dw[s3]"
            )
        if s3_client_kwargs is None:
            s3_client_kwargs = {}
        self._client = S3Client(bucket_name, **s3_client_kwargs)

    def upload_file(self, local_path: Union[str, Path], key: str, **kwargs) -> None:
        self._client.upload_file(str(local_path), key, **kwargs)

    def upload_files(self, file_mappings: dict[str, str], **kwargs) -> None:
        self._client.upload_files(file_mappings, **kwargs)

    def download_file(self, key: str, local_path: Union[str, Path], **kwargs) -> None:
        self._client.download_file(key, str(local_path), **kwargs)

    def download_files(self, file_mappings: dict[str, str], **kwargs) -> None:
        self._client.download_files(file_mappings, **kwargs)

    def list_objects(self, prefix: str) -> list[str]:
        return self._client.list_objects(prefix)

    def path_exists(self, key: str) -> bool:
        return self._client.path_exists(key)
