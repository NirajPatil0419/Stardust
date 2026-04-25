import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration
from pathlib import Path
import csv
from config.model_config import CHECKPOINT_DIR, TRAINING_LOG_DIR, MAX_DEBUG_STEPS

from config.model_config import (
    WHISPER_MODEL_NAME,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    ENGLISH_STANDARDIZED_MANIFEST,
)
from src.training.datasets.feature_dataset import WhisperFeatureDataset
from src.training.utils.feature_collate_fn import whisper_feature_collate_fn

HEARTBEAT_INTERVAL = 60  # 60 seconds = 1 minute

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

    print("Loading dataset...")
    dataset = WhisperFeatureDataset(
        manifest_path=ENGLISH_STANDARDIZED_MANIFEST,
        language="en",
        task="transcribe",
    )

    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=whisper_feature_collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ✅ CREATE FOLDERS + LOG FILE HERE
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TRAINING_LOG_DIR).mkdir(parents=True, exist_ok=True)

    log_file = Path(TRAINING_LOG_DIR) / "training_loss_log.csv"

    with open(log_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "step", "loss", "step_time_seconds"])

    training_start_time = time.time()
    print("Starting training loop...")

    last_heartbeat_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        for step, batch in enumerate(dataloader):
            step_start = time.time()

            # 🔁 Heartbeat check
            current_time = time.time()
            if current_time - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                print(f"[HEARTBEAT] Training still running... Epoch {epoch}, Step {step}")
                last_heartbeat_time = current_time

            print(f"\n[STEP {step}] Loading batch...")

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            print(f"[STEP {step}] Forward pass...")
            outputs = model(
                input_features=input_features,
                labels=labels,
            )

            loss = outputs.loss

            print(f"[STEP {step}] Backpropagation...")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step_end = time.time()

            print(
                f"[STEP {step}] DONE | Loss: {loss.item():.4f} | Time: {step_end - step_start:.2f}s"
            )
            with open(log_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, step, loss.item(), round(step_end - step_start, 2)])
            

            # Debug stop
            if step == MAX_DEBUG_STEPS:
                total_time = time.time() - training_start_time
                avg_time = total_time / (step + 1)

                print("\nStopping early after debug steps")
                print(f"Total training time: {total_time:.2f} seconds")
                print(f"Average time per step: {avg_time:.2f} seconds")

                checkpoint_path = Path(CHECKPOINT_DIR) / "whisper_debug_checkpoint"

                print(f"Saving model checkpoint to: {checkpoint_path}")
                model.save_pretrained(checkpoint_path)

                print("Checkpoint saved successfully")
                return


if __name__ == "__main__":
    main()