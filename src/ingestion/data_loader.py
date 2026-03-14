from pathlib import Path
import pandas as pd

# Root data folders
DATA_ROOT = Path("data")
INTERIM_ROOT = DATA_ROOT / "interim"
METADATA_ROOT = DATA_ROOT / "metadata"

# Supported audio formats
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}


def find_audio_files(language_folder: Path) -> list[Path]:
    """
    Recursively search for all audio files inside a language folder.

    Why this function is needed:
    Some datasets do not store audio files directly in the root folder.
    They may be inside nested folders. This function searches all
    subfolders and returns every audio file it finds.
    """
    audio_files = []

    # rglob("*") searches recursively through all folders and files
    for file_path in language_folder.rglob("*"):
        # Check only files with supported audio extensions
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(file_path)

    return audio_files


def find_metadata_files(language_folder: Path) -> list[Path]:
    """
    Find metadata files such as CSV or TSV.

    Why this function is needed:
    Datasets often store transcript mapping information in CSV/TSV files.
    We need to detect those files before we can connect transcript text
    with actual audio file names.
    """
    metadata_files = []

    for file_path in language_folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".csv", ".tsv"}:
            metadata_files.append(file_path)

    return metadata_files


def build_audio_inventory(language: str) -> pd.DataFrame:
    """
    Build a table of all audio files for one language.

    Output columns:
    - language
    - file_name
    - file_path
    - extension

    Why this function is needed:
    A structured dataframe is easier to inspect, validate, save, and join
    later with transcript metadata.
    """
    language_folder = INTERIM_ROOT / language

    if not language_folder.exists():
        raise FileNotFoundError(f"Language folder not found: {language_folder}")

    audio_files = find_audio_files(language_folder)

    rows = []
    for audio_file in audio_files:
        rows.append(
            {
                "language": language,
                "file_name": audio_file.name,
                "file_path": str(audio_file.resolve()),
                "extension": audio_file.suffix.lower(),
            }
        )

    return pd.DataFrame(rows)


def inspect_language_dataset(language: str) -> None:
    """
    Print a summary of the dataset structure for one language.

    Why this function is needed:
    Before building a full preprocessing pipeline, we need to understand
    what exists inside each dataset folder.
    """
    language_folder = INTERIM_ROOT / language

    print(f"\nInspecting language folder: {language_folder}")

    audio_files = find_audio_files(language_folder)
    metadata_files = find_metadata_files(language_folder)

    print(f"Total audio files found: {len(audio_files)}")
    print(f"Total metadata files found: {len(metadata_files)}")

    print("\nSample audio files:")
    for file_path in audio_files[:10]:
        print(f" - {file_path}")

    print("\nMetadata files:")
    for file_path in metadata_files:
        print(f" - {file_path}")


def save_audio_inventory(language: str) -> None:
    """
    Save discovered audio file inventory to CSV.

    Why this function is needed:
    This gives us a reusable file for analysis, validation, and joining
    with transcript metadata later.
    """
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)

    df = build_audio_inventory(language)
    output_file = METADATA_ROOT / f"{language}_audio_inventory.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved inventory for {language} to: {output_file}")
    print(f"Total rows saved: {len(df)}")


if __name__ == "__main__":
    # Inspect both datasets first
    inspect_language_dataset("english")
    inspect_language_dataset("hindi")

    # Save basic audio inventory files
    save_audio_inventory("english")
    save_audio_inventory("hindi")