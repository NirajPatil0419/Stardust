# src/training/train_whisper.py

import sys
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration

from config.model_config import (
    WHISPER_MODEL_NAME,
    BATCH_SIZE,
    ENGLISH_STANDARDIZED_MANIFEST,
    HINDI_STANDARDIZED_MANIFEST,
)
from src.training.datasets.feature_dataset import WhisperFeatureDataset
from src.training.utils.feature_collate_fn import whisper_feature_collate_fn


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("train_whisper.py started")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
    model.to(device)
    model.train()

    # Load a small English dataset
    english_dataset = WhisperFeatureDataset(
        manifest_path=ENGLISH_STANDARDIZED_MANIFEST,
        language="en",
        task="transcribe",
    )

    english_loader = DataLoader(
        english_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=whisper_feature_collate_fn,
    )

    # Get one batch
    batch = next(iter(english_loader))

    input_features = batch["input_features"].to(device)
    labels = batch["labels"].to(device)

    print("Batch loaded successfully")
    print("Input features shape:", input_features.shape)
    print("Labels shape:", labels.shape)

    # Forward pass
    outputs = model(
        input_features=input_features,
        labels=labels,
    )

    loss = outputs.loss
    print("Forward pass successful")
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()