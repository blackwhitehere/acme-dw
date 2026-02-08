from dataclasses import dataclass
from typing import Union, List, Optional
import os
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl

from acme_dw.backends import StorageBackend, LocalBackend


@dataclass
class DatasetMetadata:
    """Captures useful dataset metadata:
    Name of dataset source: e.g.`yahoo_finance`
    Name of the dataset: e.g. `price_history`
    Dataset version identifier: e.g. `v1`
    Unique identifier of a process that populates the dataset e.g. `fetch_yahoo_data`
    Any number of partitions specific to a dataset, e.g. `minute, AAPL, 2025`
    Name of file object: e.g. `20250124`
    Type of data stored in write object: e.g. `parquet`
    Type of DataFrame to read: e.g. `pandas` or `polars`
    """

    source: str
    name: str
    version: str
    process_id: str
    partitions: list[str]
    file_name: str
    file_type: str
    df_type: str = "pandas"

    @classmethod
    def from_dict(cls, data: dict):
        """Create a DatasetMetadata instance from a dictionary"""
        return cls(**data)


@dataclass
class DatasetPrefix:
    source: str
    name: str
    version: str
    process_id: str
    partitions: list[str]  # can be a subset of partitions in DatasetMetadata
    file_type: str = "parquet"
    df_type: str = "pandas"


class DW:
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        path_prefix: str = "dw",
        s3_client_kwargs: dict = None,
        backend: Optional[StorageBackend] = None,
    ):
        """Initialize DW client for managing data warehouse storage.

        Accepts either an explicit ``backend`` or falls back to S3 (legacy
        behaviour).  When ``backend`` is supplied, ``bucket_name`` and
        ``s3_client_kwargs`` are ignored.

        Args:
            bucket_name: Name of the S3 bucket (used when no backend given).
            path_prefix: Prefix for all paths in the data warehouse. Defaults to "dw".
            s3_client_kwargs: Optional kwargs to pass to S3Client initialization.
            backend: A StorageBackend instance.  When provided, S3 args are ignored.
        """
        if backend is not None:
            self._backend = backend
        else:
            # Legacy S3 path
            from acme_dw.backends import S3Backend

            if s3_client_kwargs is None:
                s3_client_kwargs = {}
            if bucket_name is None:
                bucket_name = os.environ["DW_AWS_BUCKET_NAME"]
            self._backend = S3Backend(bucket_name, s3_client_kwargs)
        self.path_prefix = path_prefix

    # Keep backward-compatible property so existing code referencing
    # dw.s3_client still works when using S3Backend.
    @property
    def s3_client(self):
        from acme_dw.backends import S3Backend

        if isinstance(self._backend, S3Backend):
            return self._backend._client
        raise AttributeError("s3_client is only available with S3Backend")

    @s3_client.setter
    def s3_client(self, value):
        from acme_dw.backends import S3Backend

        if isinstance(self._backend, S3Backend):
            self._backend._client = value
        else:
            raise AttributeError("s3_client is only available with S3Backend")

    def _get_key(self, metadata: DatasetMetadata):
        return f"{self.path_prefix}/{metadata.source}/{metadata.name}/{metadata.version}/{metadata.process_id}/{'/'.join(metadata.partitions)}/{metadata.file_name}.{metadata.file_type}"

    # Keep old name for backward compatibility
    _get_s3_key = _get_key

    def write_df(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        metadata: Union[DatasetMetadata, dict],
        to_parquet_kwargs: dict = None,
        s3_kwargs: dict = None,
    ):
        """Write a DataFrame to storage as a parquet file with metadata.

        Args:
            df: Pandas DataFrame or Polars DataFrame to write
            metadata: DatasetMetadata object or dict containing metadata
            to_parquet_kwargs: Optional kwargs to pass to to_parquet()
            s3_kwargs: Optional kwargs to pass to storage upload
        """
        if to_parquet_kwargs is None:
            to_parquet_kwargs = {}
        if s3_kwargs is None:
            s3_kwargs = {}

        if isinstance(df, pl.DataFrame):
            to_parquet_func = df.write_parquet
        elif isinstance(df, pd.DataFrame):
            to_parquet_func = df.to_parquet
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(df)}")

        if isinstance(metadata, dict):
            metadata = DatasetMetadata.from_dict(metadata)

        key = self._get_key(metadata)

        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            to_parquet_func(tmp.name, **to_parquet_kwargs)
            self._backend.upload_file(tmp.name, key, **s3_kwargs)

    def write_many_dfs(
        self,
        df_list: list[Union[pd.DataFrame, pl.DataFrame]],
        metadata_list: List[Union[DatasetMetadata, dict]],
        to_parquet_kwargs: dict = None,
        s3_kwargs: dict = None,
    ):
        """Write multiple DataFrames to storage as parquet files.

        Args:
            df_list: List of DataFrames to write
            metadata_list: List of DatasetMetadata objects or dicts
            to_parquet_kwargs: Optional kwargs to pass to to_parquet()
            s3_kwargs: Optional kwargs to pass to storage upload
        """
        if to_parquet_kwargs is None:
            to_parquet_kwargs = {}
        if s3_kwargs is None:
            s3_kwargs = {}

        if isinstance(df_list[0], pl.DataFrame):
            to_parquet_func = df_list[0].write_parquet
        elif isinstance(df_list[0], pd.DataFrame):
            to_parquet_func = df_list[0].to_parquet
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(df_list[0])}")

        file_mappings = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (df, metadata) in enumerate(zip(df_list, metadata_list)):
                tmp_path = Path(tmpdir) / f"file_{i}.parquet"
                to_parquet_func(tmp_path, **to_parquet_kwargs)
                if isinstance(metadata, dict):
                    metadata = DatasetMetadata.from_dict(metadata)
                key = self._get_key(metadata)
                file_mappings[str(tmp_path)] = key

            self._backend.upload_files(file_mappings, **s3_kwargs)

    def read_df(
        self,
        metadata: Union[DatasetMetadata, dict],
        read_parquet_kwargs: dict = None,
        s3_kwargs: dict = None,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Read a single DataFrame from the data warehouse.

        Args:
            metadata: DatasetMetadata object or dict describing the dataset.
            read_parquet_kwargs: Optional kwargs to pass to read_parquet()
            s3_kwargs: Optional kwargs to pass to storage download

        Returns:
            pandas or polars DataFrame loaded from the parquet file.
        """
        if read_parquet_kwargs is None:
            read_parquet_kwargs = {}
        if s3_kwargs is None:
            s3_kwargs = {}

        if isinstance(metadata, dict):
            metadata = DatasetMetadata.from_dict(metadata)

        if metadata.df_type == "pandas":
            read_parquet_func = pd.read_parquet
        elif metadata.df_type == "polars":
            read_parquet_func = pl.read_parquet
        else:
            raise ValueError(f"Unsupported DataFrame type: {metadata.df_type}")

        key = self._get_key(metadata)
        if not self._backend.path_exists(key):
            raise FileNotFoundError(f"File not found: {key}")

        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            self._backend.download_file(key, tmp.name, **s3_kwargs)
            return read_parquet_func(tmp.name, **read_parquet_kwargs)

    def _get_key_prefix(self, dataset_prefix: DatasetPrefix) -> str:
        return f"{self.path_prefix}/{dataset_prefix.source}/{dataset_prefix.name}/{dataset_prefix.version}/{dataset_prefix.process_id}/{'/'.join(dataset_prefix.partitions)}"

    # Keep old name for backward compatibility
    _get_s3_key_prefix = _get_key_prefix

    def read_dataset(
        self,
        dataset_prefix: DatasetPrefix,
        read_parquet_kwargs: dict = None,
        s3_kwargs: dict = None,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Read a dataset from multiple parquet files under a prefix.

        Args:
            dataset_prefix: DatasetPrefix describing what to read.
            read_parquet_kwargs: Optional kwargs to pass to read_parquet()
            s3_kwargs: Optional kwargs to pass to storage download

        Returns:
            Combined pandas or polars DataFrame.
        """
        if read_parquet_kwargs is None:
            read_parquet_kwargs = {}
        if s3_kwargs is None:
            s3_kwargs = {}

        if dataset_prefix.df_type == "pandas":
            read_parquet_func = pd.read_parquet
        elif dataset_prefix.df_type == "polars":
            read_parquet_func = pl.read_parquet
        else:
            raise ValueError(f"Unsupported DataFrame type: {dataset_prefix.df_type}")

        key_prefix = self._get_key_prefix(dataset_prefix)
        keys = self._backend.list_objects(key_prefix)
        keys = [k for k in keys if k.endswith(dataset_prefix.file_type)]

        if len(keys) == 0:
            raise FileNotFoundError(f"No files found with prefix: {key_prefix}")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_mappings = {
                key: Path(tmpdir) / Path(key.replace(key_prefix + "/", ""))
                for key in keys
            }
            self._backend.download_files(file_mappings, **s3_kwargs)
            return read_parquet_func(tmpdir, **read_parquet_kwargs)
