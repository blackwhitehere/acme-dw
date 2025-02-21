# acme-dw

Simple data warehouse using S3

# Problem

Some LLM based definitions:
`A data warehouse is a centralized repository designed for storing, managing, and analyzing structured data from various sources, optimized for query performance and reporting. It typically uses a schema-based approach to organize data in tables and supports complex queries and analytics. In contrast, a data lake is a storage system that holds vast amounts of raw, unstructured, and structured data in its native format until needed. It is designed for scalability and flexibility, allowing for the storage of diverse data types and enabling advanced analytics, machine learning, and big data processing.`

We can see how S3 can be easily utilized as a data lake with little extra functionality. However to use it as a data warehouse we need to add some extra functionality that largly depends on the needs of a given domain.

# Features

* Provides read/wrie on schema-less `pd.DataFrame`
* Saves `pd.DataFrame` using parquet format for fast read performance.
* Standardizes metadata associated with each dataset

# Dev environment

The project comes with a python development environment.
To generate it, after checking out the repo run:

    chmod +x create_env.sh

Then to generate the environment (or update it to latest version based on state of `uv.lock`), run:

    ./create_env.sh

This will generate a new python virtual env under `.venv` directory. You can activate it via:

    source .venv/bin/activate

If you are using VSCode, set to use this env via `Python: Select Interpreter` command.

## Example usage

    from acme_dw import DW, DatasetMetadata

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
    df = dw.read_df(metadata)

# Project template

This project has been setup with `acme-project-create`, a python code template library.

# Required setup post use

* Enable GitHub Pages to be published via [GitHub Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) by going to `Settings-->Pages-->Source`
* Create `release-pypi` environment for [GitHub Actions](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-deployments/managing-environments-for-deployment#creating-an-environment) to enable uploads of the library to PyPi
* Setup auth to PyPI for the GitHub Action implemented in `.github/workflows/release.yml` via [Trusted Publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) `uv publish` [doc](https://docs.astral.sh/uv/guides/publish/#publishing-your-package)
* Once you create the python environment for the first time add the `uv.lock` file that will be created in project directory to the source control and update it each time environment is rebuilt
* In order not to replicate documentation in `docs/docs/index.md` file and `README.md` in root of the project setup a symlink from `README.md` file to the `index.md` file.
To do this, from `docs/docs` dir run:

    ln -sf ../../README.md index.md