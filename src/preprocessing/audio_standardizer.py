from pathlib import Path
import pandas as pd
import librosa
import soundfile as sf

# Root folders
DATA_ROOT = Path("data")
PROCESSED_ROOT = DATA_ROOT / "processed"
STANDARDIZED_ROOT = DATA_ROOT / "standardized"

# Target audio standard for training
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_EXTENSION = ".wav"


def standardize_audio_file(input_path: str, output_path: str) -> bool:
    """
    Convert one audio file into a standard format:
    - mono
    - 16 kHz
    - WAV

    Why this function is needed:
    AI speech models perform better when every file has the same
    sample rate and channel configuration. This reduces errors
    during feature extraction and training.

    Returns:
        True  -> conversion successful
        False -> conversion failed
    """
    try:
        # Load audio and resample to target sample rate.
        # mono=True forces mono conversion.
        audio, sr = librosa.load(input_path, sr=TARGET_SAMPLE_RATE, mono=True)

        # Ensure parent folder exists before writing file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as WAV using soundfile
        sf.write(output_path, audio, TARGET_SAMPLE_RATE)

        return True

    except Exception as error:
        print(f"Failed to standardize file: {input_path}")
        print(f"Error: {error}")
        return False


def build_standardized_output_path(language: str, split: str, audio_file: str) -> Path:
    """
    Create a clean output path for standardized audio.

    Example output:
    data/standardized/english/train/sample-000001.wav

    Why this function is needed:
    We want a predictable folder structure for training and debugging.
    """
    # Extract only the final file name from paths like:
    # cv-valid-train/sample-000001.mp3
    file_stem = Path(audio_file).stem

    output_path = STANDARDIZED_ROOT / language / split / f"{file_stem}{TARGET_EXTENSION}"
    return output_path


def standardize_manifest(language: str) -> pd.DataFrame:
    """
    Standardize all audio files for one language based on manifest_clean.csv.

    Input:
        data/processed/<language>/manifest_clean.csv

    Output:
        data/processed/<language>/manifest_standardized.csv
    """
    # input_manifest = PROCESSED_ROOT / language / "manifest_clean.csv"
    input_manifest = PROCESSED_ROOT / language / "manifest_sample.csv"

    if not input_manifest.exists():
        raise FileNotFoundError(f"Clean manifest not found: {input_manifest}")

    df = pd.read_csv(input_manifest)

    print(f"\nStarting audio standardization for language: {language}")
    print(f"Total rows in clean manifest: {len(df)}")

    standardized_rows = []
    success_count = 0
    fail_count = 0

    for _, row in df.iterrows():
        input_audio_path = row["audio_path"]
        split = row["split"]
        audio_file = row["audio_file"]

        output_audio_path = build_standardized_output_path(language, split, audio_file)

        success = standardize_audio_file(
            input_path=input_audio_path,
            output_path=str(output_audio_path)
        )

        if success:
            success_count += 1

            standardized_row = row.to_dict()
            standardized_row["standardized_audio_path"] = str(output_audio_path.resolve())
            standardized_row["target_sample_rate"] = TARGET_SAMPLE_RATE
            standardized_row["target_channels"] = TARGET_CHANNELS
            standardized_rows.append(standardized_row)
        else:
            fail_count += 1

    print(f"Successful conversions for {language}: {success_count}")
    print(f"Failed conversions for {language}: {fail_count}")

    return pd.DataFrame(standardized_rows)


def save_standardized_manifest(language: str) -> None:
    """
    Save the final standardized manifest for one language.
    """
    output_manifest = PROCESSED_ROOT / language / "manifest_standardized.csv"

    standardized_df = standardize_manifest(language)
    standardized_df.to_csv(output_manifest, index=False)

    print(f"Saved standardized manifest: {output_manifest}")
    print(f"Final standardized rows for {language}: {len(standardized_df)}")


if __name__ == "__main__":
    save_standardized_manifest("english")
    save_standardized_manifest("hindi")