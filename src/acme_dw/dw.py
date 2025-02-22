from dataclasses import dataclass
from typing import Union, List
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
from acme_s3 import S3Client


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


class DW:
    def __init__(self, bucket_name: str, path_prefix: str = "dw", s3_client_kwargs: dict = None):
        """Initialize DW client for managing data warehouse on S3

        Args:
            bucket_name: Name of the S3 bucket to use as data warehouse
            path_prefix: Prefix for all paths in the data warehouse. Defaults to "dw"
            s3_client_kwargs: Optional kwargs to pass to S3Client initialization
        """
        if s3_client_kwargs is None:
            s3_client_kwargs = {}
        self.s3_client = S3Client(bucket_name, **s3_client_kwargs)
        self.path_prefix = path_prefix

    def _get_s3_key(self, metadata: DatasetMetadata):
        return f"{self.path_prefix}/{metadata.source}/{metadata.name}/{metadata.version}/{metadata.process_id}/{'/'.join(metadata.partitions)}/{metadata.file_name}.{metadata.file_type}"

    def write_df(
        self,
        df: Union[pd.DataFrame, pl.DataFrame],
        metadata: Union[DatasetMetadata, dict],
        to_parquet_kwargs: dict = None,
        s3_kwargs: dict = None,
    ):
        """Write a pandas DataFrame to S3 as a parquet file with metadata

        Args:
            df: Pandas DataFrame or Polars DataFrame to write
            metadata: DatasetMetadata object or dict containing metadata
            to_parquet_kwargs: Optional kwargs to pass to pandas to_parquet()
            s3_kwargs: Optional kwargs to pass to S3 upload

        Example:
            ```python
            dw = DW('my-bucket')
            
            # Write with DatasetMetadata object
            metadata = DatasetMetadata(
                source='yahoo_finance',
                name='price_history', 
                version='v1',
                process_id='fetch_yahoo_data',
                partitions=['minute', 'AAPL', '2025'],
                file_name='20250124',
                file_type='parquet'
            )
            dw.write_df(df, metadata)

            # Write with metadata dict
            metadata_dict = {
                'source': 'yahoo_finance',
                'name': 'price_history',
                'version': 'v1', 
                'process_id': 'fetch_yahoo_data',
                'partitions': ['minute', 'AAPL', '2025'],
                'file_name': '20250124',
                'file_type': 'parquet'
            }
            dw.write_df(df, metadata_dict)
            ```
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

        # Convert dict to DatasetMetadata if needed
        if isinstance(metadata, dict):
            metadata = DatasetMetadata.from_dict(metadata)

        # Get the S3 key for this dataset
        s3_key = self._get_s3_key(metadata)

        # Create a temporary file and write DataFrame as parquet
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            to_parquet_func(tmp.name, **to_parquet_kwargs)
            # Upload the temporary file to S3
            self.s3_client.upload_file(tmp.name, s3_key, **s3_kwargs)

    def write_many_dfs(
        self,
        df_list: list[Union[pd.DataFrame, pl.DataFrame]],
        metadata_list: List[Union[DatasetMetadata, dict]],
        to_parquet_kwargs: dict = None,
        s3_kwargs: dict = None,
    ):
        """Write multiple pandas DataFrames to S3 as parquet files with metadata

        Args:
            df_list: List of pandas DataFrames or Polars DataFrames to write
            metadata_list: List of DatasetMetadata objects or dicts containing metadata
            to_parquet_kwargs: Optional kwargs to pass to pandas to_parquet()
            s3_kwargs: Optional kwargs to pass to S3 upload

        Example:
            ```python
            dw = DW('my-bucket')
            # Write multiple DataFrames with metadata dicts
            metadata_list = [
                {
                    'source': 'yahoo_finance',
                    'name': 'price_history', 
                    'version': 'v1',
                    'process_id': 'fetch_yahoo_data',
                    'partitions': ['minute', 'AAPL', '2025'],
                    'file_name': '20250124',
                    'file_type': 'parquet'
                },
                {
                    'source': 'yahoo_finance', 
                    'name': 'price_history',
                    'version': 'v1',
                    'process_id': 'fetch_yahoo_data', 
                    'partitions': ['minute', 'MSFT', '2025'],
                    'file_name': '20250124',
                    'file_type': 'parquet'
                }
            ]
            dw.write_many_dfs([df1, df2], metadata_list)
            ```
            
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

        # Create temporary files and build mapping
        file_mappings = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (df, metadata) in enumerate(zip(df_list, metadata_list)):
                tmp_path = Path(tmpdir) / f"file_{i}.parquet"
                to_parquet_func(tmp_path, **to_parquet_kwargs)
                # Convert dict to DatasetMetadata if needed
                if isinstance(metadata, dict):
                    metadata = DatasetMetadata.from_dict(metadata)
                s3_key = self._get_s3_key(metadata)
                file_mappings[str(tmp_path)] = s3_key

            # Upload all files in parallel
            self.s3_client.upload_files(file_mappings, **s3_kwargs)

    def read_df(
        self,
        metadata: Union[DatasetMetadata, dict],
        read_parquet_kwargs: dict = None,
        s3_kwargs: dict = None,
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Read a single DataFrame from the data warehouse

        This method downloads a parquet file from S3 based on the provided metadata and loads it into a pandas DataFrame.

        Args:
            metadata: Either a DatasetMetadata object or a dictionary containing metadata about the dataset to read.
                     Must include source, name, version, process_id, partitions, file_name and file_type.
            read_parquet_kwargs: Optional dictionary of keyword arguments to pass to pandas.read_parquet()
            s3_kwargs: Optional dictionary of keyword arguments to pass to S3 download operation

        Returns:
            pandas.DataFrame or polars.DataFrame: The DataFrame loaded from the parquet file

        Example:
            ```python
            metadata = {
                'source': 'yahoo_finance',
                'name': 'price_history',
                'version': 'v1', 
                'process_id': 'fetch_yahoo_data',
                'partitions': ['minute', 'AAPL', '2025'],
                'file_name': '20250124',
                'file_type': 'parquet'
            }
            df = dw.read_df(metadata)
            ```
        """
        if read_parquet_kwargs is None:
            read_parquet_kwargs = {}
        if s3_kwargs is None:
            s3_kwargs = {}

        if metadata.df_type == "pandas":
            read_parquet_func = pd.read_parquet
        elif metadata.df_type == "polars":
            read_parquet_func = pl.read_parquet
        else:
            raise ValueError(f"Unsupported DataFrame type: {metadata.df_type}")

        # Convert dict to DatasetMetadata if needed
        if isinstance(metadata, dict):
            metadata = DatasetMetadata.from_dict(metadata)

        # Get the S3 key for this dataset
        s3_key = self._get_s3_key(metadata)

        # Create a temporary file to download to
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            # Download the file from S3
            self.s3_client.download_file(s3_key, tmp.name, **s3_kwargs)
            # Read the parquet file into a DataFrame
            return read_parquet_func(tmp.name, **read_parquet_kwargs)

if __name__ == "__main__":
    import os
    dw = DW(os.environ['TEST_AWS_BUCKET_NAME'], path_prefix='dw-test')
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    metadata = DatasetMetadata(
        source='sample_source',
        name='sample_dataset',
        version='v1',
        process_id='sample_process',
        partitions=['partition1', 'partition2'],
        file_name='sample_file',
        file_type='parquet'
    )
    print(f"Uploading df to S3 with metadata:\n{metadata}")
    print(df)
    dw.write_df(df, metadata)
    print(f"Downloading df from S3 with metadata:\n{metadata}")
    df_read = dw.read_df(metadata)
    print(df_read)
