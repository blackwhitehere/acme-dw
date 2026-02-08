import pytest
import pandas as pd
from pathlib import Path

from acme_dw.backends import LocalBackend
from acme_dw.dw import DW, DatasetMetadata, DatasetPrefix


@pytest.fixture
def local_root(tmp_path):
    return tmp_path / "dw_store"


@pytest.fixture
def backend(local_root):
    return LocalBackend(local_root)


@pytest.fixture
def local_dw(backend):
    return DW(backend=backend, path_prefix="dw")


@pytest.fixture
def sample_metadata():
    return DatasetMetadata(
        source="test_source",
        name="test_dataset",
        version="v1",
        process_id="test_process",
        partitions=["part1"],
        file_name="file1",
        file_type="parquet",
        df_type="pandas",
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})


# --- LocalBackend unit tests ---


def test_upload_and_download(backend, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    backend.upload_file(src, "foo/bar.txt")
    assert backend.path_exists("foo/bar.txt")

    dest = tmp_path / "dest.txt"
    backend.download_file("foo/bar.txt", dest)
    assert dest.read_text() == "hello"


def test_upload_files(backend, tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("aaa")
    f2.write_text("bbb")
    backend.upload_files({str(f1): "dir/a.txt", str(f2): "dir/b.txt"})
    assert backend.path_exists("dir/a.txt")
    assert backend.path_exists("dir/b.txt")


def test_download_files(backend, tmp_path):
    src = tmp_path / "orig.txt"
    src.write_text("data")
    backend.upload_file(src, "k1.txt")

    dest = tmp_path / "out" / "k1.txt"
    backend.download_files({"k1.txt": str(dest)})
    assert dest.read_text() == "data"


def test_list_objects(backend, tmp_path):
    for name in ["a.parquet", "b.parquet", "c.csv"]:
        src = tmp_path / name
        src.write_text("x")
        backend.upload_file(src, f"prefix/{name}")

    keys = backend.list_objects("prefix")
    assert len(keys) == 3
    assert "prefix/a.parquet" in keys


def test_path_exists_false(backend):
    assert not backend.path_exists("nonexistent/key")


def test_download_file_not_found(backend, tmp_path):
    with pytest.raises(FileNotFoundError):
        backend.download_file("missing.txt", tmp_path / "out.txt")


# --- Integration: DW + LocalBackend ---


def test_roundtrip_write_read(local_dw, sample_metadata, sample_df):
    local_dw.write_df(sample_df, sample_metadata)
    result = local_dw.read_df(sample_metadata)
    pd.testing.assert_frame_equal(result, sample_df)


def test_roundtrip_polars(local_dw, sample_df):
    import polars as pl

    meta = DatasetMetadata(
        source="src",
        name="ds",
        version="v1",
        process_id="proc",
        partitions=["p1"],
        file_name="f1",
        file_type="parquet",
        df_type="polars",
    )
    pl_df = pl.from_pandas(sample_df)
    local_dw.write_df(pl_df, meta)
    result = local_dw.read_df(meta)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == pl_df.shape


def test_write_many_and_read_dataset(local_dw):
    df1 = pd.DataFrame({"x": [1, 2]})
    df2 = pd.DataFrame({"x": [3, 4]})
    m1 = DatasetMetadata(
        source="s",
        name="d",
        version="v1",
        process_id="p",
        partitions=["p1"],
        file_name="f1",
        file_type="parquet",
    )
    m2 = DatasetMetadata(
        source="s",
        name="d",
        version="v1",
        process_id="p",
        partitions=["p1"],
        file_name="f2",
        file_type="parquet",
    )
    local_dw.write_many_dfs([df1, df2], [m1, m2])

    prefix = DatasetPrefix(
        source="s",
        name="d",
        version="v1",
        process_id="p",
        partitions=["p1"],
        file_type="parquet",
        df_type="pandas",
    )
    result = local_dw.read_dataset(prefix)
    assert len(result) == 4


def test_read_df_file_not_found(local_dw, sample_metadata):
    with pytest.raises(FileNotFoundError):
        local_dw.read_df(sample_metadata)


def test_read_dataset_no_files(local_dw):
    prefix = DatasetPrefix(
        source="s",
        name="d",
        version="v1",
        process_id="p",
        partitions=["missing"],
        file_type="parquet",
        df_type="pandas",
    )
    with pytest.raises(FileNotFoundError, match="No files found"):
        local_dw.read_dataset(prefix)
