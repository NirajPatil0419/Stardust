# config/model_config.py

"""
Central configuration file for model and training settings.

Why this file exists:
Instead of hardcoding model names and hyperparameters across many files,
we keep them here in one place. This makes the project cleaner,
easier to update, and more professional.
"""

# Hugging Face Whisper model to use
WHISPER_MODEL_NAME = "openai/whisper-small"

# Language settings
ENGLISH_LANGUAGE_CODE = "en"
HINDI_LANGUAGE_CODE = "hi"

# Whisper task options:
# - transcribe  -> speech to text in same language
# - translate   -> speech translated to English
WHISPER_TASK = "transcribe"

# Training hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1

# Audio settings
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Paths
ENGLISH_STANDARDIZED_MANIFEST = "data/processed/english/manifest_standardized.csv"
HINDI_STANDARDIZED_MANIFEST = "data/processed/hindi/manifest_standardized.csv"