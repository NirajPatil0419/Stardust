import sys
from torch.utils.data import DataLoader
from src.training.datasets.audio_dataset import AudioTranslationDataset
from src.training.utils.collate_fn import audio_collate_fn


def main():
    """
    Test the dataset loader with a custom collate function.

    Why this script is needed:
    Audio samples have variable lengths, so we must use a custom collate
    function to pad them before batching.
    """

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    english_manifest = "data/processed/english/manifest_standardized.csv"
    hindi_manifest = "data/processed/hindi/manifest_standardized.csv"

    english_dataset = AudioTranslationDataset(english_manifest)
    hindi_dataset = AudioTranslationDataset(hindi_manifest)

    print(f"English dataset size: {len(english_dataset)}")
    print(f"Hindi dataset size: {len(hindi_dataset)}")

    # Use custom collate function
    english_loader = DataLoader(
        english_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=audio_collate_fn
    )

    hindi_loader = DataLoader(
        hindi_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=audio_collate_fn
    )

    # Test English batch
    print("\nTesting English loader...")
    english_batch = next(iter(english_loader))
    print("English batch keys:", english_batch.keys())
    print("English padded waveform shape:", english_batch["padded_waveforms"].shape)
    print("English waveform lengths:", english_batch["waveform_lengths"])
    print("English transcripts:", english_batch["transcripts"])

    # Test Hindi batch
    print("\nTesting Hindi loader...")
    hindi_batch = next(iter(hindi_loader))
    print("Hindi batch keys:", hindi_batch.keys())
    print("Hindi padded waveform shape:", hindi_batch["padded_waveforms"].shape)
    print("Hindi waveform lengths:", hindi_batch["waveform_lengths"])
    print("Hindi transcripts:", hindi_batch["transcripts"])


if __name__ == "__main__":
    main()