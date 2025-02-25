import pytest
import pandas as pd
import polars as pl
from unittest.mock import patch, ANY
from acme_dw.dw import DW, DatasetMetadata, DatasetPrefix

@pytest.fixture
def mock_s3_client():
    """Fixture that provides a mocked S3Client"""
    with patch('acme_dw.dw.S3Client') as mock:
        yield mock.return_value

@pytest.fixture
def dw(mock_s3_client):
    """Fixture that provides a DW instance with mocked S3Client"""
    dw = DW('test-bucket', path_prefix='test-dw')
    dw.s3_client = mock_s3_client
    return dw

@pytest.fixture
def sample_metadata():
    """Fixture that provides a sample DatasetMetadata object"""
    return DatasetMetadata(
        source='test_source',
        name='test_dataset',
        version='v1',
        process_id='test_process',
        partitions=['part1', 'part2'],
        file_name='test_file',
        file_type='parquet',
        df_type='pandas'
    )

@pytest.fixture
def sample_metadata_polars():
    """Fixture that provides a sample DatasetMetadata object for polars"""
    return DatasetMetadata(
        source='test_source',
        name='test_dataset',
        version='v1',
        process_id='test_process',
        partitions=['part1', 'part2'],
        file_name='test_file',
        file_type='parquet',
        df_type='polars'
    )

@pytest.fixture
def sample_prefix():
    """Fixture that provides a sample DatasetPrefix object"""
    return DatasetPrefix(
        source='test_source',
        name='test_dataset',
        version='v1',
        process_id='test_process',
        partitions=['part1'],
        file_type='parquet',
        df_type='pandas'
    )

@pytest.fixture
def sample_pandas_df():
    """Fixture that provides a sample pandas DataFrame"""
    return pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

@pytest.fixture
def sample_polars_df():
    """Fixture that provides a sample polars DataFrame"""
    return pl.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

def test_write_df_pandas(dw, sample_metadata, sample_pandas_df):
    """Tests that write_df correctly uploads a pandas DataFrame to S3"""
    dw.write_df(sample_pandas_df, sample_metadata)
    
    dw.s3_client.upload_file.assert_called_once()

def test_write_df_polars(dw, sample_metadata_polars, sample_polars_df):
    """Tests that write_df correctly uploads a polars DataFrame to S3"""
    dw.write_df(sample_polars_df, sample_metadata_polars)
    
    dw.s3_client.upload_file.assert_called_once()

def test_write_df_with_dict_metadata(dw, sample_pandas_df):
    """Tests that write_df correctly handles metadata provided as a dictionary"""
    metadata_dict = {
        'source': 'test_source',
        'name': 'test_dataset',
        'version': 'v1',
        'process_id': 'test_process',
        'partitions': ['part1', 'part2'],
        'file_name': 'test_file',
        'file_type': 'parquet',
        'df_type': 'pandas'
    }
    
    dw.write_df(sample_pandas_df, metadata_dict)
    
    dw.s3_client.upload_file.assert_called_once()

@patch('pandas.DataFrame.to_parquet')
def test_write_many_dfs(mock_to_parquet, dw, sample_pandas_df, sample_metadata):
    """Tests that write_many_dfs correctly uploads multiple DataFrames to S3"""    
    dfs = [sample_pandas_df] * 2
    metadata_list = [sample_metadata] * 2
    
    # This should build file mappings with temp files
    dw.write_many_dfs(dfs, metadata_list)
    
    # Check that to_parquet was called twice (for each DataFrame)
    assert mock_to_parquet.call_count == 2
    
    # Check that upload_files was called with a dict mapping
    dw.s3_client.upload_files.assert_called_once()

@patch('pandas.read_parquet')
def test_read_df_pandas(mock_pd_read, dw, sample_metadata):
    """Tests that read_df correctly downloads and reads a pandas DataFrame from S3"""
    dw.s3_client.path_exists.return_value = True
    # Set up mock to return a DataFrame
    mock_pd_read.return_value = pd.DataFrame({'test': [1, 2, 3]})
    
    result = dw.read_df(sample_metadata)
    
    dw.s3_client.download_file.assert_called_once()
    # Check pandas read_parquet was called with temp file
    mock_pd_read.assert_called_once()
    # Verify the return value is the DataFrame from read_parquet
    assert isinstance(result, pd.DataFrame)

@patch('polars.read_parquet')
def test_read_df_polars(mock_pl_read, dw, sample_metadata_polars):
    """Tests that read_df correctly downloads and reads a polars DataFrame from S3"""
    dw.s3_client.path_exists.return_value = True
    # Set up mock to return a DataFrame
    mock_pl_read.return_value = pl.DataFrame({'test': [1, 2, 3]})
    
    result = dw.read_df(sample_metadata_polars)
    
    dw.s3_client.download_file.assert_called_once()
    mock_pl_read.assert_called_once()
    # Verify the return value is the DataFrame from read_parquet
    assert isinstance(result, pl.DataFrame)

def test_read_df_file_not_found(dw, sample_metadata):
    """Tests that read_df raises FileNotFoundError when the file doesn't exist in S3"""
    dw.s3_client.path_exists.return_value = False
    
    with pytest.raises(FileNotFoundError, match=f".*{sample_metadata.file_name}.*"):
        dw.read_df(sample_metadata)

@patch('pandas.read_parquet')
def test_read_dataset(mock_pd_read, dw, mock_s3_client, sample_prefix):
    """Tests that read_dataset correctly downloads and reads a dataset from S3"""
    # Mock list_objects to return some file paths
    dw.s3_client.list_objects.return_value = [
        'test-dw/test_source/test_dataset/v1/test_process/part1/part2/file1.parquet',
        'test-dw/test_source/test_dataset/v1/test_process/part1/part2/file2.parquet'
    ]
    # Setup return value for read_parquet
    mock_pd_read.return_value = pd.DataFrame({'test': [1, 2, 3]})
    
    result = dw.read_dataset(sample_prefix)
    
    # Check S3 operations
    mock_s3_client.list_objects.assert_called_once()
    mock_s3_client.download_files.assert_called_once()
    # Check that pandas read_parquet was called
    mock_pd_read.assert_called_once()
    # Verify the return value is the DataFrame from read_parquet
    assert isinstance(result, pd.DataFrame)

def test_read_dataset_no_files(dw, mock_s3_client, sample_prefix):
    """Tests that read_dataset raises FileNotFoundError when no files are found"""
    mock_s3_client.list_objects.return_value = []
    
    with pytest.raises(FileNotFoundError, match="No files found"):
        dw.read_dataset(sample_prefix)

def test_unsupported_df_type(dw, sample_metadata):
    """Tests that read_df raises ValueError when unsupported df_type is provided"""
    sample_metadata.df_type = 'unsupported'
    
    with pytest.raises(ValueError, match="Unsupported DataFrame type"):
        dw.read_df(sample_metadata)

def test_get_s3_key_prefix(dw, sample_prefix):
    """Tests that _get_s3_key_prefix correctly formats the prefix path"""
    expected = 'test-dw/test_source/test_dataset/v1/test_process/part1'
    assert dw._get_s3_key_prefix(sample_prefix) == expected
