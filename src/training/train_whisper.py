import sys
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration

from config.model_config import (
    WHISPER_MODEL_NAME,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    ENGLISH_STANDARDIZED_MANIFEST,
)
from src.training.datasets.feature_dataset import WhisperFeatureDataset
from src.training.utils.feature_collate_fn import whisper_feature_collate_fn


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("train_whisper.py started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
    model.to(device)
    model.train()

    # Dataset
    dataset = WhisperFeatureDataset(
        manifest_path=ENGLISH_STANDARDIZED_MANIFEST,
        language="en",
        task="transcribe",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=whisper_feature_collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training loop...")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        for step, batch in enumerate(dataloader):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_features=input_features,
                labels=labels,
            )

            loss = outputs.loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

            # IMPORTANT: stop early (we are just testing)
            if step == 20:
                print("Stopping early after 20 steps (test run)")
                return


if __name__ == "__main__":
    main()