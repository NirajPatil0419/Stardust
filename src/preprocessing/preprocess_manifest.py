from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data")
PROCESSED_ROOT = DATA_ROOT / "processed"


def normalize_text(text: str) -> str:
    """
    Clean transcript text by:
    - converting to string
    - trimming spaces
    - removing line breaks
    - collapsing repeated spaces
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text


def preprocess_manifest(language: str) -> pd.DataFrame:
    """
    Read a manifest, clean it, validate it, and return a cleaned dataframe.
    """
    input_file = PROCESSED_ROOT / language / "manifest.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {input_file}")

    df = pd.read_csv(input_file)

    print(f"\nProcessing language: {language}")
    print(f"Initial rows: {len(df)}")

    # Drop rows with missing critical columns
    df = df.dropna(subset=["audio_path", "transcript"])
    print(f"Rows after dropping null audio_path/transcript: {len(df)}")

    # Normalize transcript text
    df["transcript"] = df["transcript"].apply(normalize_text)

    # Remove rows where transcript becomes empty after cleaning
    df = df[df["transcript"] != ""]
    print(f"Rows after removing empty transcripts: {len(df)}")

    # Validate file existence
    df["file_exists"] = df["audio_path"].apply(lambda x: Path(x).exists())
    df = df[df["file_exists"]]
    print(f"Rows after validating audio files exist: {len(df)}")

    # Remove exact duplicates
    df = df.drop_duplicates(subset=["audio_path", "transcript"])
    print(f"Rows after removing duplicates: {len(df)}")

    # Add simple quality metrics
    df["transcript_length"] = df["transcript"].apply(len)
    df["word_count"] = df["transcript"].apply(lambda x: len(x.split()))

    # Optional: remove very short transcripts like 1-character junk
    df = df[df["transcript_length"] >= 2]
    print(f"Rows after removing extremely short transcripts: {len(df)}")

    # Drop helper column
    df = df.drop(columns=["file_exists"])

    return df


def save_clean_manifest(language: str) -> None:
    """
    Save cleaned manifest for one language.
    """
    output_file = PROCESSED_ROOT / language / "manifest_clean.csv"
    cleaned_df = preprocess_manifest(language)
    cleaned_df.to_csv(output_file, index=False)

    print(f"Saved cleaned manifest: {output_file}")
    print(f"Final cleaned rows for {language}: {len(cleaned_df)}")


if __name__ == "__main__":
    save_clean_manifest("english")
    save_clean_manifest("hindi")