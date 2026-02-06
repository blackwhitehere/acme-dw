# acme-dw

Layer 3 (Data Infrastructure) library. S3-based data warehouse — read/write Pandas and Polars DataFrames as Parquet files to S3.

## Build and Test

```bash
uv sync                      # Install dependencies
uv run pytest tests/ -v      # Run tests
```

**Note**: This is a legacy project — no `justfile`, no `ruff` linting configured. Tests may require S3 access (via `acme-s3`) or mocking.

## Architecture

- `src/acme_dw/dw.py` — Core warehouse operations (read/write DataFrames as Parquet)
- `src/acme_dw/_main.py` — CLI entry point (`adw` command)

## Project Conventions

- CLI: `adw` (entry point in pyproject.toml)
- Dependencies: `acme_s3>=0.0.5`, `pandas`, `polars`, `pyarrow`
- Supports both Pandas and Polars DataFrames for read/write
- Downstream users: `acme-dm`, `acme-prefect`
- `admin/` has `refresh_credentials.sh` and `test_s3_access.sh` for AWS credential setup
