import argparse
import os
import sys
from typing import List, Tuple

DATASET_ID = "jp797498e/twitter-entity-sentiment-analysis"
REQUIRED_FILES = ["twitter_training.csv", "twitter_validation.csv"]


def download_dataset() -> str:
    """Download the dataset via kagglehub and return the local folder path."""
    try:
        import kagglehub  # type: ignore
    except ModuleNotFoundError:
        print(
            "ERROR: kagglehub is not installed. Install it with: `uv add kagglehub` or `pip install kagglehub`.",
            file=sys.stderr,
        )
        sys.exit(1)

    path = kagglehub.dataset_download(DATASET_ID)
    return path


def verify_dataset_folder(path: str) -> Tuple[bool, List[str]]:
    """Verify that the expected CSV files exist inside the given folder.

    Returns a tuple of (is_valid, missing_files).
    """
    missing: List[str] = []
    for filename in REQUIRED_FILES:
        if not os.path.exists(os.path.join(path, filename)):
            missing.append(filename)
    return (len(missing) == 0, missing)


def write_env_variable(
    env_file_path: str, variable_name: str, value: str, overwrite: bool
) -> None:
    """Create or update an entry in a .env file.

    If overwrite is False and the variable already exists, the function will exit with code 1.
    """
    lines: List[str] = []
    if os.path.exists(env_file_path):
        with open(env_file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    variable_line_prefix = f"{variable_name}="
    existing_index = next(
        (i for i, l in enumerate(lines) if l.startswith(variable_line_prefix)), -1
    )

    new_line = f'{variable_name}="{value}"'

    if existing_index >= 0:
        if not overwrite:
            print(
                f"ERROR: {variable_name} already exists in {env_file_path}. Use --overwrite to replace it.",
                file=sys.stderr,
            )
            sys.exit(1)
        lines[existing_index] = new_line
    else:
        lines.append(new_line)

    with open(env_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Twitter Entity Sentiment Analysis dataset via kagglehub and optionally set DATA_FOLDER in a .env file.",
    )
    parser.add_argument(
        "--set-env",
        action="store_true",
        help="Write DATA_FOLDER to a .env file after download.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file to create/update (default: .env).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite DATA_FOLDER in the .env file if it already exists.",
    )
    args = parser.parse_args()

    dataset_path = download_dataset()
    print(f"Dataset downloaded to: {dataset_path}")

    is_valid, missing = verify_dataset_folder(dataset_path)
    if not is_valid:
        print(
            "ERROR: The downloaded folder is missing expected files: "
            + ", ".join(missing)
            + f".\nChecked folder: {dataset_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.set_env:
        env_file_path = os.path.abspath(args.env_file)
        write_env_variable(env_file_path, "DATA_FOLDER", dataset_path, args.overwrite)
        print(f'Updated {env_file_path} with DATA_FOLDER="{dataset_path}"')
    else:
        print(
            "Set DATA_FOLDER in your .env to the path above (ensure it contains twitter_training.csv and twitter_validation.csv)."
        )


if __name__ == "__main__":
    main()
