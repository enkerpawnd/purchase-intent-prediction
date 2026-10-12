"""Load the unchanged source CSV without preprocessing or fitting a model."""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "online_shoppers_intention.csv"
)


def load_data(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read a CSV; explicit relative paths use the caller's working directory."""
    path = Path(data_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {path}. See data/raw/README.md for dataset setup."
        )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV to load (default: the repository's data/raw source CSV).",
    )
    args = parser.parse_args()
    try:
        data = load_data(args.data_path)
    except (
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        parser.exit(status=1, message=f"Unable to load CSV: {exc}\n")
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns.")
    print(data.head().to_string(index=False))


if __name__ == "__main__":
    main()
