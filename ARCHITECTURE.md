# Project Architecture

## Overview

This document explains the data architecture, file responsibilities, and workflow for the `Stardust` project. It is written for beginners and uses a simple diagram plus step-by-step descriptions.

The system is a multilingual audio transcription pipeline with the following major stages:

1. **Data ingestion**
2. **Manifest creation**
3. **Transcript cleaning**
4. **Audio standardization**
5. **Dataset loading**
6. **Model training**

---

## Mermaid Architecture Diagram

```mermaid
flowchart LR
    A[Raw Audio Files<br/>(data/interim/)] --> B[Inventory Files<br/>(data/metadata/*_audio_inventory.csv)]
    A --> C[Metadata CSV/TSV]
    B --> D[Manifest Builder<br/>src/preprocessing/training_manifest.py]
    C --> D
    D --> E[Processed Manifest<br/>(data/processed/<language>/manifest.csv)]
    E --> F[Clean Manifest<br/>(data/processed/<language>/manifest_clean.csv)]
    F --> G[Standardized Audio Files<br/>(data/standardized/<language>/<split>/)]
    G --> H[Standardized Manifest<br/>(data/processed/<language>/manifest_standardized.csv)]
    H --> I[Whisper Feature Dataset<br/>src/training/datasets/feature_dataset.py]
    I --> J[Training Script<br/>src/training/train_whisper.py]
    I --> K[Feature Collate Function<br/>src/training/utils/feature_collate_fn.py]
    J --> L[Hugging Face Transformers / Whisper]
    J --> M[Config<br/>config/model_config.py]

    subgraph Ingestion
        B
        C
        D
    end
    subgraph Preprocessing
        E
        F
        G
        H
    end
    subgraph Training
        I
        J
        K
        L
    end

    A -->|discovered by| B
    B -->|used by| D
    C -->|used by| D
    D -->|creates| E
    E -->|cleanup step| F
    F -->|standardize audio| G
    G -->|records path in| H
    H -->|loaded by| I
    I -->|batched with| K
    J -->|loads dataset| I
    J -->|uses config| M
    J -->|loads model from| L
```

---

## Module and Class Connections

This project does not use Kaggle in code. The only external model system connected at runtime is Hugging Face via the `transformers` library and the Whisper model.

- `src/preprocessing/training_manifest.py` is the main manifest builder. It reads raw metadata CSV/TSV files and resolves audio file paths into a standard manifest.
- `src/ingestion/data_loader.py` is a helper module to find audio files and metadata files. It is not directly imported by `training_manifest.py` in the current repo, but it shows how ingestion can discover files automatically.
- `src/preprocessing/preprocess_manifest.py` reads the manifest created by `training_manifest.py`, cleans transcripts, and removes invalid rows.
- `src/preprocessing/audio_standardizer.py` reads the clean manifest (or sample manifest), converts audio to 16kHz mono WAV, and writes standardized audio files.
- `src/training/datasets/feature_dataset.py` reads the final standardized manifest and transforms audio and text into model-ready tensors.
- `src/training/utils/processor_loader.py` loads the Hugging Face Whisper processor from the model name defined in `config/model_config.py`.
- `src/training/train_whisper.py` loads the Whisper model and the dataset, then trains using batches from `src/training/utils/feature_collate_fn.py`.

### File-level dependencies

- `src/training/train_whisper.py` imports:
  - `WhisperForConditionalGeneration` from `transformers`
  - `WhisperFeatureDataset` from `src/training/datasets/feature_dataset.py`
  - `whisper_feature_collate_fn` from `src/training/utils/feature_collate_fn.py`
  - configuration values from `config/model_config.py`
- `src/training/datasets/feature_dataset.py` imports:
  - `load_whisper_processor` from `src/training/utils/processor_loader.py`
  - `WhisperProcessor` indirectly via `processor_loader.py`
- `src/training/utils/processor_loader.py` imports:
  - `WhisperProcessor` from `transformers`
  - `WHISPER_MODEL_NAME` from `config/model_config.py`
- `src/training/utils/feature_collate_fn.py` pads feature and label tensors produced by `WhisperFeatureDataset`.

---

## Data Folder Structure

The project stores data in the `data/` folder using these subfolders:

- `data/raw/` - original source datasets, not modified by the pipeline
- `data/interim/` - extracted dataset files organized by language
- `data/metadata/` - manifests and inventory files created during ingestion
- `data/processed/` - cleaned manifest CSV files and processed metadata
- `data/standardized/` - converted audio files ready for model training

### Language support

The project includes at least two languages:
- `english`
- `hindi`

Each language has its own processed manifests and standardized audio.

---

## What Data the Project Has

The main data artifacts are:

- `audio files` - raw speech recordings in formats such as `.mp3`, `.wav`, `.flac`, `.m4a`
- `metadata files` - CSV or TSV records that map audio file references to transcripts
- `manifest.csv` - compiled mapping of audio path, language, transcript, and split
- `manifest_sample.csv` or `manifest_clean.csv` - cleaned subset ready for standardization
- `manifest_standardized.csv` - final manifest with standardized audio path and audio metadata

---

## File-by-File Purpose and Usage

### `src/ingestion/data_loader.py`
- Purpose: find audio files and metadata files inside language folders
- Uses: `data/interim/<language>/`
- Output: discovered file paths used by manifest creation
- Why: raw datasets are inconsistent and may store audio in nested directories

### `src/preprocessing/training_manifest.py`
- Purpose: build manifests for each language from metadata files
- Reads: metadata files in `data/interim/<language>/`
- Writes:
  - `data/metadata/<language>_manifest.csv`
  - `data/processed/<language>/manifest.csv`
- Why: the model needs one canonical manifest that connects audio paths and transcripts

### `src/preprocessing/preprocess_manifest.py`
- Purpose: clean and validate manifest text data
- Reads: `data/processed/<language>/manifest.csv`
- Writes: cleaned manifest in memory or a later save step
- Why: transcripts may include invalid text, line breaks, or empty rows that break training

### `src/preprocessing/audio_standardizer.py`
- Purpose: convert audio to a consistent format
- Reads: `data/processed/<language>/manifest_sample.csv` (or `manifest_clean.csv`)
- Writes:
  - standardized audio files under `data/standardized/<language>/<split>/`
  - `data/processed/<language>/manifest_standardized.csv`
- Why: audio models require consistent sample rate, channel count, and file format

### `config/model_config.py`
- Purpose: central configuration for model names, hyperparameters, and manifest paths
- Contains: `WHISPER_MODEL_NAME`, language codes, task name, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`, `TARGET_SAMPLE_RATE`, `TARGET_CHANNELS`, and standardized manifest paths
- Why: this file gives a single location where the training model and data paths are configured

### `src/training/datasets/audio_dataset.py`
- Purpose: load raw waveform audio and transcripts from a standardized manifest
- Reads: `data/processed/<language>/manifest_standardized.csv`
- Returns: one sample at a time with waveform tensor and text
- Why: training loops need a PyTorch `Dataset` object, not raw CSV files

### `src/training/datasets/feature_dataset.py`
- Purpose: create Whisper-ready training inputs
- Reads: `data/processed/<language>/manifest_standardized.csv`
- Uses: Whisper processor to convert audio into input features and transcripts into label IDs
- Returns: model-ready tensors for input and label training
- Why: Whisper training requires preprocessed audio features and tokenized labels

### `src/training/utils/processor_loader.py`
- Purpose: load the Hugging Face Whisper processor
- Uses: `config/model_config.py` for the model identifier
- Why: the processor handles audio featurization and tokenizer creation for Whisper

### `src/training/utils/collate_fn.py`
- Purpose: batch variable-length raw waveforms safely
- Used by: `test_dataset_loader.py` and any DataLoader that loads raw audio samples
- Why: PyTorch cannot stack waveforms of different lengths without padding

### `src/training/utils/feature_collate_fn.py`
- Purpose: batch Whisper feature tensors and label sequences
- Used by: `WhisperFeatureDataset` training DataLoaders
- Why: input feature lengths and label lengths vary across samples

### `src/training/train_whisper.py`
- Purpose: main training script for Whisper
- Uses:
  - `WhisperForConditionalGeneration`
  - `WhisperFeatureDataset`
  - `whisper_feature_collate_fn`
- Why: it orchestrates model loading, data loading, training loop, and optimization

### `src/training/test_dataset_loader.py`
- Purpose: verify dataset loading and waveform batching
- Runs: a small sample through `AudioTranslationDataset`
- Why: ensures preprocessing and batching work correctly before training

### `src/training/test_feature_dataset.py`
- Purpose: verify Whisper feature dataset loading and batching
- Runs: a small sample through `WhisperFeatureDataset`
- Why: checks the end-to-end path from standardized audio to model-ready tensors

---

## Data Flow Summary

1. **Raw audio and metadata** enter the pipeline from `data/interim/<language>/`.
2. `training_manifest.py` reads metadata and resolves audio file references.
3. It writes a canonical manifest in `data/processed/<language>/manifest.csv`.
4. `preprocess_manifest.py` cleans transcripts and removes invalid rows.
5. `audio_standardizer.py` converts audio to 16 kHz mono WAV and saves standardized files under `data/standardized/`.
6. It also writes `data/processed/<language>/manifest_standardized.csv` with final audio paths.
7. `WhisperFeatureDataset` loads this standardized manifest and builds training tensors.
8. `train_whisper.py` uses the data loader and Whisper model to run training.

---

## How the Data Is Used

- `audio_path` from the manifest points to raw or original audio.
- `standardized_audio_path` points to the converted audio ready for training.
- `transcript` is cleaned and used as the target text during training.
- `language` and `split` are metadata fields used for organization and debugging.
- `target_sample_rate` and `target_channels` are added to the manifest after standardization so downstream code can verify audio consistency.

---

## Why This Architecture Exists

- **Modularity**: each step has one clear responsibility.
- **Reproducibility**: manifests record exactly which files were used.
- **Robustness**: audio standardization and manifest cleaning reduce downstream failure.
- **Scalability**: new languages or datasets can be added by following the same folder conventions.

---

## Beginner Tips

- Start by exploring `data/interim/` to see raw audio and metadata structure.
- Open `data/processed/<language>/manifest.csv` to inspect the dataset mapping.
- Run the test scripts first to confirm the data pipeline works before training.
- If `config/model_config.py` is empty, add model and training parameters there and update imports.

---

## Recommended Next Step

If you want to run the pipeline, make sure the following exist:
- `data/interim/english/` and `data/interim/hindi/`
- metadata CSV/TSV files for each language
- a populated `config/model_config.py` with model constants

Then follow the file order in this document to understand how each stage transforms the data.
