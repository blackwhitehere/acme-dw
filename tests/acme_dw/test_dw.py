import pytest
import pandas as pd
import polars as pl
from unittest.mock import patch, MagicMock
from acme_dw.dw import DW, DatasetMetadata, DatasetPrefix
from acme_dw.backends import S3Backend


@pytest.fixture
def mock_s3_backend():
    """Fixture that provides a DW with a mocked S3Backend."""
    backend = MagicMock(spec=S3Backend)
    backend._client = MagicMock()
    return backend


@pytest.fixture
def dw(mock_s3_backend):
    """Fixture that provides a DW instance with mocked S3Backend."""
    return DW(backend=mock_s3_backend, path_prefix="test-dw")


@pytest.fixture
def sample_metadata():
    return DatasetMetadata(
        source="test_source",
        name="test_dataset",
        version="v1",
        process_id="test_process",
        partitions=["part1", "part2"],
        file_name="test_file",
        file_type="parquet",
        df_type="pandas",
    )


@pytest.fixture
def sample_metadata_polars():
    return DatasetMetadata(
        source="test_source",
        name="test_dataset",
        version="v1",
        process_id="test_process",
        partitions=["part1", "part2"],
        file_name="test_file",
        file_type="parquet",
        df_type="polars",
    )


@pytest.fixture
def sample_prefix():
    return DatasetPrefix(
        source="test_source",
        name="test_dataset",
        version="v1",
        process_id="test_process",
        partitions=["part1"],
        file_type="parquet",
        df_type="pandas",
    )


@pytest.fixture
def sample_pandas_df():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@pytest.fixture
def sample_polars_df():
    return pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


def test_write_df_pandas(dw, sample_metadata, sample_pandas_df):
    dw.write_df(sample_pandas_df, sample_metadata)
    dw._backend.upload_file.assert_called_once()


def test_write_df_polars(dw, sample_metadata_polars, sample_polars_df):
    dw.write_df(sample_polars_df, sample_metadata_polars)
    dw._backend.upload_file.assert_called_once()


def test_write_df_with_dict_metadata(dw, sample_pandas_df):
    metadata_dict = {
        "source": "test_source",
        "name": "test_dataset",
        "version": "v1",
        "process_id": "test_process",
        "partitions": ["part1", "part2"],
        "file_name": "test_file",
        "file_type": "parquet",
        "df_type": "pandas",
    }
    dw.write_df(sample_pandas_df, metadata_dict)
    dw._backend.upload_file.assert_called_once()


@patch("pandas.DataFrame.to_parquet")
def test_write_many_dfs(mock_to_parquet, dw, sample_pandas_df, sample_metadata):
    dfs = [sample_pandas_df] * 2
    metadata_list = [sample_metadata] * 2
    dw.write_many_dfs(dfs, metadata_list)
    assert mock_to_parquet.call_count == 2
    dw._backend.upload_files.assert_called_once()


@patch("pandas.read_parquet")
def test_read_df_pandas(mock_pd_read, dw, sample_metadata):
    dw._backend.path_exists.return_value = True
    mock_pd_read.return_value = pd.DataFrame({"test": [1, 2, 3]})
    result = dw.read_df(sample_metadata)
    dw._backend.download_file.assert_called_once()
    mock_pd_read.assert_called_once()
    assert isinstance(result, pd.DataFrame)


@patch("polars.read_parquet")
def test_read_df_polars(mock_pl_read, dw, sample_metadata_polars):
    dw._backend.path_exists.return_value = True
    mock_pl_read.return_value = pl.DataFrame({"test": [1, 2, 3]})
    result = dw.read_df(sample_metadata_polars)
    dw._backend.download_file.assert_called_once()
    mock_pl_read.assert_called_once()
    assert isinstance(result, pl.DataFrame)


def test_read_df_file_not_found(dw, sample_metadata):
    dw._backend.path_exists.return_value = False
    with pytest.raises(FileNotFoundError, match=f".*{sample_metadata.file_name}.*"):
        dw.read_df(sample_metadata)


@patch("pandas.read_parquet")
def test_read_dataset(mock_pd_read, dw, sample_prefix):
    dw._backend.list_objects.return_value = [
        "test-dw/test_source/test_dataset/v1/test_process/part1/part2/file1.parquet",
        "test-dw/test_source/test_dataset/v1/test_process/part1/part2/file2.parquet",
    ]
    mock_pd_read.return_value = pd.DataFrame({"test": [1, 2, 3]})
    result = dw.read_dataset(sample_prefix)
    dw._backend.list_objects.assert_called_once()
    dw._backend.download_files.assert_called_once()
    mock_pd_read.assert_called_once()
    assert isinstance(result, pd.DataFrame)


def test_read_dataset_no_files(dw, sample_prefix):
    dw._backend.list_objects.return_value = []
    with pytest.raises(FileNotFoundError, match="No files found"):
        dw.read_dataset(sample_prefix)


def test_unsupported_df_type(dw, sample_metadata):
    sample_metadata.df_type = "unsupported"
    with pytest.raises(ValueError, match="Unsupported DataFrame type"):
        dw.read_df(sample_metadata)


def test_get_s3_key_prefix(dw, sample_prefix):
    expected = "test-dw/test_source/test_dataset/v1/test_process/part1"
    assert dw._get_key_prefix(sample_prefix) == expected
    # backward compat alias
    assert dw._get_s3_key_prefix(sample_prefix) == expected
