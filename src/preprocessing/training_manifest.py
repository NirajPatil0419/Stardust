from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data")
INTERIM_ROOT = DATA_ROOT / "interim"
METADATA_ROOT = DATA_ROOT / "metadata"
PROCESSED_ROOT = DATA_ROOT / "processed"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}


def find_audio_files(language_folder: Path) -> list[Path]:
    audio_files = []

    for file_path in language_folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(file_path)

    return audio_files


def build_audio_lookup(language: str):
    """
    Build multiple lookup styles because datasets are inconsistent.

    We store:
    1. exact relative path from the language root
    2. last two path parts, like folder/file.mp3
    3. plain filename, only for fallback
    """
    language_folder = INTERIM_ROOT / language
    audio_files = find_audio_files(language_folder)

    exact_relative_lookup = {}
    short_relative_lookup = {}
    filename_lookup = {}

    for audio_file in audio_files:
        resolved_path = str(audio_file.resolve())

        # Example:
        # cv-valid-train/cv-valid-train/sample-000000.mp3
        relative_path = audio_file.relative_to(language_folder).as_posix()
        exact_relative_lookup[relative_path] = resolved_path

        # Example:
        # cv-valid-train/sample-000000.mp3
        parts = Path(relative_path).parts
        if len(parts) >= 2:
            short_key = "/".join(parts[-2:])
            short_relative_lookup[short_key] = resolved_path

        # Example:
        # sample-000000.mp3
        file_name = audio_file.name
        if file_name not in filename_lookup:
            filename_lookup[file_name] = []
        filename_lookup[file_name].append(resolved_path)

    return exact_relative_lookup, short_relative_lookup, filename_lookup


def read_metadata_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix.lower() == ".tsv":
        return pd.read_csv(file_path, sep="\t")

    raise ValueError(f"Unsupported metadata file: {file_path}")


def detect_schema_columns(df: pd.DataFrame):
    columns = set(df.columns)

    if "filename" in columns and "text" in columns:
        return "filename", "text"

    if "path" in columns and "sentence" in columns:
        return "path", "sentence"

    return None, None


def infer_split_from_file_name(metadata_file: Path) -> str:
    name = metadata_file.stem.lower()

    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "test" in name:
        return "test"
    if "valid" in name:
        return "validated"
    if "invalid" in name:
        return "invalidated"
    if "other" in name:
        return "other"

    return name


def resolve_audio_path(audio_reference, exact_relative_lookup, short_relative_lookup, filename_lookup):
    """
    Try multiple ways to find the real audio file path.
    """
    audio_reference = str(audio_reference).strip().replace("\\", "/")

    # 1. Exact match from metadata
    if audio_reference in exact_relative_lookup:
        return exact_relative_lookup[audio_reference]

    # 2. Match using folder/file.mp3 form
    if audio_reference in short_relative_lookup:
        return short_relative_lookup[audio_reference]

    # 3. Match using just file name if unique
    file_name = Path(audio_reference).name
    candidates = filename_lookup.get(file_name, [])

    if len(candidates) == 1:
        return candidates[0]

    return None


def build_manifest_for_language(language: str) -> pd.DataFrame:
    language_folder = INTERIM_ROOT / language
    exact_relative_lookup, short_relative_lookup, filename_lookup = build_audio_lookup(language)

    manifest_rows = []

    for metadata_file in language_folder.rglob("*"):
        if not metadata_file.is_file():
            continue
        if metadata_file.suffix.lower() not in {".csv", ".tsv"}:
            continue

        df = read_metadata_file(metadata_file)
        audio_col, text_col = detect_schema_columns(df)

        # skip non-training metadata like durations/reported/etc
        if not audio_col or not text_col:
            continue

        split_name = infer_split_from_file_name(metadata_file)

        for _, row in df.iterrows():
            audio_reference = str(row[audio_col]).strip()
            transcript = str(row[text_col]).strip()

            full_audio_path = resolve_audio_path(
                audio_reference,
                exact_relative_lookup,
                short_relative_lookup,
                filename_lookup,
            )

            if full_audio_path and transcript:
                manifest_rows.append(
                    {
                        "language": language,
                        "split": split_name,
                        "audio_file": audio_reference,
                        "audio_path": full_audio_path,
                        "transcript": transcript,
                    }
                )

    return pd.DataFrame(manifest_rows)


def save_manifest(language: str) -> None:
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    (PROCESSED_ROOT / language).mkdir(parents=True, exist_ok=True)

    df = build_manifest_for_language(language)

    metadata_output = METADATA_ROOT / f"{language}_manifest.csv"
    processed_output = PROCESSED_ROOT / language / "manifest.csv"

    df.to_csv(metadata_output, index=False)
    df.to_csv(processed_output, index=False)

    print(f"{language} manifest rows: {len(df)}")
    print(f"Saved metadata manifest: {metadata_output}")
    print(f"Saved processed manifest: {processed_output}")


if __name__ == "__main__":
    save_manifest("english")
    save_manifest("hindi")