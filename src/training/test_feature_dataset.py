import sys
from torch.utils.data import DataLoader

from src.training.datasets.feature_dataset import WhisperFeatureDataset
from src.training.utils.feature_collate_fn import whisper_feature_collate_fn


def main():
    print("test_feature_dataset.py started")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    english_manifest = "data/processed/english/manifest_standardized.csv"
    hindi_manifest = "data/processed/hindi/manifest_standardized.csv"

    print("Loading datasets...")

    english_dataset = WhisperFeatureDataset(
        manifest_path=english_manifest,
        language="en",
        task="transcribe",
    )

    hindi_dataset = WhisperFeatureDataset(
        manifest_path=hindi_manifest,
        language="hi",
        task="transcribe",
    )

    print(f"English feature dataset size: {len(english_dataset)}")
    print(f"Hindi feature dataset size: {len(hindi_dataset)}")

    english_loader = DataLoader(
        english_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=whisper_feature_collate_fn,
    )

    hindi_loader = DataLoader(
        hindi_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=whisper_feature_collate_fn,
    )

    print("\nTesting English feature loader...")
    english_batch = next(iter(english_loader))
    print("English batch keys:", english_batch.keys())
    print("English input_features shape:", english_batch["input_features"].shape)
    print("English labels shape:", english_batch["labels"].shape)
    print("English transcripts:", english_batch["transcripts"])

    print("\nTesting Hindi feature loader...")
    hindi_batch = next(iter(hindi_loader))
    print("Hindi batch keys:", hindi_batch.keys())
    print("Hindi input_features shape:", hindi_batch["input_features"].shape)
    print("Hindi labels shape:", hindi_batch["labels"].shape)
    print("Hindi transcripts:", hindi_batch["transcripts"])


if __name__ == "__main__":
    main()