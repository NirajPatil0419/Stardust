# Stardust: AI Audio Transcription and Translation Pipeline

## Presentation purpose

This document explains the Stardust project in presentation-friendly language. It covers:

- what the project does;
- how data flows through the system;
- what each important file does;
- which AI, machine-learning, and NLP concepts are used;
- what the six logged training steps mean; and
- what is implemented today versus the next-stage roadmap.

---

## 1. Executive summary

Stardust is a local Python machine-learning project for preparing multilingual speech data and fine-tuning OpenAI Whisper for speech-to-text.

The project currently works with English and Hindi audio. It takes an audio recording and its matching transcript, cleans and standardizes the data, converts it into Whisper-ready numerical inputs, and runs a small training job to verify that the full pipeline works.

The small sample is intentional. Training a full audio dataset on a local machine can take a very long time, so the project uses sample manifests to validate the code, data format, model loading, batching, loss calculation, logging, validation, and checkpoint saving before moving to larger compute.

In one sentence:

> Stardust is a supervised deep-learning pipeline that teaches a pretrained Whisper model to map speech audio to text.

---

## 2. The main problem it solves

Audio datasets are usually inconsistent:

- recordings can be MP3, WAV, FLAC, or M4A;
- files can have different sample rates and mono/stereo channels;
- transcript metadata can use different column names;
- recordings and transcripts may live in nested folders; and
- some rows can have missing files, blank transcripts, or duplicates.

Whisper training needs a reliable mapping between each audio file and its correct text. Stardust turns messy source data into a predictable training dataset.

---

## 3. High-level architecture

~~~mermaid
flowchart LR
    A[Audio recordings and transcript metadata] --> B[Build audio/transcript manifest]
    B --> C[Clean manifest]
    C --> D[Standardize audio: mono, 16 kHz, WAV]
    D --> E[Standardized manifest]
    E --> F[Whisper feature dataset]
    F --> G[Batch and pad samples]
    G --> H[Fine-tune Whisper]
    H --> I[Loss log, validation output, model checkpoint]

    J[Separate English-to-Hindi text translator] -. independent demo .-> K[MarianMT model]
~~~

The main training path is:

    data/interim/<language>
    -> manifest.csv
    -> manifest_clean.csv
    -> manifest_sample.csv
    -> standardized WAV files + manifest_standardized.csv
    -> Whisper features and token labels
    -> training and checkpoint

---

## 4. Key terms for the audience

| Term | Simple meaning in Stardust |
|---|---|
| Transcript | The correct text spoken in an audio recording. |
| Manifest | A CSV spreadsheet that connects an audio-file path to its transcript and metadata. |
| Feature | A numerical representation of data that a model can process. |
| Token | A numeric text unit used by a language model instead of raw words. |
| Batch | A small group of examples processed together during training. |
| Loss | A number that measures model error. Lower is generally better. |
| Fine-tuning | Taking a pretrained model and adapting it with more specific examples. |
| Checkpoint | A saved copy of model weights after training. |

---

## 5. Project folders

| Folder | Purpose |
|---|---|
| data/interim/ | Extracted source audio and metadata, organized by language. |
| data/metadata/ | Audio inventories and manifests created during data discovery. |
| data/processed/ | Cleaned, sampled, and standardized manifest CSV files. |
| data/standardized/ | Converted mono 16 kHz WAV recordings used for training. |
| src/ingestion/ | Code that discovers audio and metadata files. |
| src/preprocessing/ | Code that builds, cleans, and standardizes datasets. |
| src/training/ | Dataset classes, batching helpers, tests, and Whisper training code. |
| src/translation/ | A separate English-to-Hindi text translation demo. |
| config/ | Central model and training settings. |
| logs/training/ | Training-loss CSV output. |
| models/checkpoints/ | Saved Whisper model checkpoint. |

---

## 6. File-by-file explanation and concepts used

| File | What it does | Concepts used | Why those concepts are used |
|---|---|---|---|
| [config/model_config.py](config/model_config.py) | Stores model name, language codes, batch size, learning rate, epochs, and data paths. | Hyperparameters, configuration management. | Keeps training choices in one place instead of hard-coding them throughout the project. |
| [src/ingestion/data_loader.py](src/ingestion/data_loader.py) | Recursively finds audio files and CSV/TSV metadata files, then can create an audio inventory. | Data ingestion, data discovery, metadata inventory. | Datasets can contain nested folders and inconsistent structures. |
| [src/preprocessing/training_manifest.py](src/preprocessing/training_manifest.py) | Matches metadata rows to real audio files and writes a canonical manifest. | Data integration, schema detection, data validation. | English and Hindi source metadata can use different column names and file-path formats. |
| [src/preprocessing/preprocess_manifest.py](src/preprocessing/preprocess_manifest.py) | Removes missing, blank, invalid, and duplicate records; cleans transcript text. | Data cleaning, basic text normalization, descriptive quality metrics. | Better input data prevents training failures and improves data quality. |
| [src/preprocessing/audio_standardizer.py](src/preprocessing/audio_standardizer.py) | Converts source audio to mono, 16 kHz WAV and creates the final standardized manifest. | Feature engineering, audio/signal preprocessing, resampling. | Whisper expects predictable audio input; one common format reduces errors. |
| [src/training/datasets/audio_dataset.py](src/training/datasets/audio_dataset.py) | Loads raw standardized waveform tensors and transcript metadata. | PyTorch Dataset abstraction, tensor conversion. | Lets PyTorch fetch one structured training item at a time. |
| [src/training/utils/collate_fn.py](src/training/utils/collate_fn.py) | Pads raw waveforms to a common length within a batch. | Variable-length sequence padding, batching. | Audio clips have different durations and cannot be stacked directly. |
| [src/training/datasets/feature_dataset.py](src/training/datasets/feature_dataset.py) | Converts WAV audio into Whisper input features and transcripts into token IDs. | Feature extraction, tokenization, encoding, NLP preprocessing. | Whisper trains on numerical audio features and token labels, not raw files or plain text. |
| [src/training/utils/processor_loader.py](src/training/utils/processor_loader.py) | Loads the Hugging Face Whisper processor. | Transfer learning, pretrained processor reuse. | The processor knows how to create features and tokenize text for the selected Whisper model. |
| [src/training/utils/feature_collate_fn.py](src/training/utils/feature_collate_fn.py) | Pads feature tensors and token-label sequences for Whisper batches. | Sequence padding, masked loss labels. | Each recording and transcript has a different length. Padding lets a batch train together while ignored labels do not affect loss. |
| [src/training/train_whisper.py](src/training/train_whisper.py) | Loads Whisper, creates DataLoaders, trains, records loss, validates, and saves a checkpoint. | Supervised learning, deep neural networks, Transformer, backpropagation, AdamW optimization, training/validation split. | This is the central learning loop that improves the model from audio/transcript examples. |
| [src/training/test_dataset_loader.py](src/training/test_dataset_loader.py) | Manually checks waveform dataset loading and padding. | Smoke testing, batch-shape verification. | Confirms data can be read before a longer training run begins. |
| [src/training/test_feature_dataset.py](src/training/test_feature_dataset.py) | Manually checks Whisper feature creation and batch shapes. | Smoke testing, feature validation, tokenization validation. | Confirms the model-ready dataset works end to end. |
| [src/translation/text_translator.py](src/translation/text_translator.py) | Runs an interactive English-to-Hindi text translator using MarianMT. | Neural machine translation, sequence-to-sequence NLP, tokenization. | Demonstrates text translation separately from the Whisper transcription pipeline. |

---

## 7. Concepts used in the project

### Directly used concepts

| Concept | How Stardust uses it |
|---|---|
| Supervised learning | Each input audio clip has a correct target transcript. |
| Feature engineering | Audio is converted to mono, 16 kHz WAV; Whisper then creates numerical audio features. |
| Deep neural network | Whisper is a pretrained Transformer neural network. |
| Transfer learning / fine-tuning | The project adapts openai/whisper-small instead of training a model from scratch. |
| Tokenization | Transcript text becomes Whisper token IDs. |
| Encoding | Audio becomes numerical features; text becomes numerical token IDs. |
| Backpropagation | The training loop runs loss.backward() to calculate gradients. |
| Optimization | AdamW changes model parameters to reduce loss. |
| Batching and padding | Two samples are trained together; varying audio/text lengths are padded. |
| Hyperparameters | Batch size, learning rate, epoch count, model name, and debug-step count are set in configuration. |
| Basic text analysis | The project cleans transcript whitespace and records transcript length and word count. |

### Used indirectly inside the pretrained model

| Concept | How it applies |
|---|---|
| Probability | Whisper assigns probabilities to possible next text tokens. |
| Categorical token distribution | At each output position, the model selects from its vocabulary using probabilities. This is the closest project concept to a multinomial-style output. |
| Cross-entropy-style token loss | The model compares predicted token probabilities with the correct transcript tokens. The resulting loss is logged during training. |
| Activation functions | Whisper uses them internally, but Stardust does not define custom activation functions. |
| Convolutional audio layers | Whisper contains internal audio-processing layers, but Stardust is not a custom CNN project. Its main architecture is a Transformer. |

### Present as configuration, but not systematic tuning

The project has hyperparameters, but it does not yet implement hyperparameter tuning.

Examples of fixed settings:

    BATCH_SIZE = 2
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 1
    MAX_DEBUG_STEPS = 5

Hyperparameter tuning would mean systematically comparing several combinations of those settings and selecting the best validation result.

---

## 8. Concepts not currently used

The following topics are not implemented directly in Stardust:

| Area | Not currently used |
|---|---|
| Probability distributions | Bernoulli, binomial, geometric, negative binomial, Poisson, hypergeometric, uniform, normal, exponential, and explicit continuous-distribution modeling. |
| Statistical hypothesis testing | Hypothesis tests, t-tests, chi-square tests, and ANOVA. |
| Regression | Linear regression and logistic regression. |
| Decision trees | CART, Gini impurity, information gain, and entropy-based tree splitting. |
| Clustering | Hierarchical clustering and distance measures such as Manhattan, Minkowski, Chebyshev, cosine, Mahalanobis, or Hamming distance. |
| Ensemble learning | Random Forest, bagging, and other ensemble techniques. |
| Conventional classification reporting | Confusion matrix and classification accuracy. Speech recognition is sequence generation, not ordinary class-label prediction. |

For speech-to-text, Word Error Rate (WER) and Character Error Rate (CER) would be more useful future evaluation metrics than a confusion matrix.

---

## 9. Training run: why there are six logged steps

The current settings are deliberately small so the local machine can validate the pipeline without training on the full dataset:

    BATCH_SIZE = 2
    MAX_DEBUG_STEPS = 5

Python counts training steps from zero. Therefore, steps 0, 1, 2, 3, 4, and 5 run: six training updates in total.

Each update processes two audio/transcript examples, so this debug run trains on 12 English samples before validation and checkpoint saving.

At each step:

    1. Load a batch of audio and transcripts.
    2. Convert audio to Whisper input features.
    3. Ask Whisper to predict transcript tokens.
    4. Calculate loss against the correct tokens.
    5. Backpropagate the error.
    6. Update weights with AdamW.
    7. Write loss and duration to the CSV log.

### Training-loss log

Source: [logs/training/training_loss_log.csv](logs/training/training_loss_log.csv)

| Epoch | Step | Loss | Step time |
|---:|---:|---:|---:|
| 1 | 0 | 4.6992 | 131.88 s |
| 1 | 1 | 4.4530 | 134.40 s |
| 1 | 2 | 6.9665 | 174.35 s |
| 1 | 3 | 3.1785 | 161.07 s |
| 1 | 4 | 5.4256 | 181.45 s |
| 1 | 5 | 3.2444 | 186.62 s |

- Total debug-run training time: about 16 minutes 10 seconds.
- Average step time: about 2 minutes 42 seconds.
- Average training loss: about 4.66.

### Loss graph

~~~mermaid
xychart-beta
    title "Whisper Debug Training Loss by Step"
    x-axis "Step" [0, 1, 2, 3, 4, 5]
    y-axis "Loss (lower is better)" 0 --> 8
    line [4.6992, 4.4530, 6.9665, 3.1785, 5.4256, 3.2444]
~~~

How to explain the graph:

- Lower loss generally means the model's predicted transcript tokens are closer to the correct transcript.
- Loss does not need to decline on every step because every batch contains different audio quality, sentence lengths, accents, and transcript difficulty.
- Step 2 is higher because that two-sample batch was likely harder for the model.
- Six steps are enough to confirm that the training pipeline works, but not enough to judge final model quality.

After step 5, the script validates with the English dev split and saves a model checkpoint. Validation loss is printed to the terminal; the CSV currently stores training loss only.

---

## 10. Generated outputs

| Output | Location | Meaning |
|---|---|---|
| Training loss log | [logs/training/training_loss_log.csv](logs/training/training_loss_log.csv) | Step-by-step training loss and duration. |
| Whisper checkpoint folder | [models/checkpoints/whisper_debug_checkpoint](models/checkpoints/whisper_debug_checkpoint) | Saved model after the debug run. |
| Model weights | [model.safetensors](models/checkpoints/whisper_debug_checkpoint/model.safetensors) | The learned parameters of the fine-tuned model. |
| Model configuration | [config.json](models/checkpoints/whisper_debug_checkpoint/config.json) | Whisper architecture/configuration information. |
| Generation configuration | [generation_config.json](models/checkpoints/whisper_debug_checkpoint/generation_config.json) | Inference/generation settings. |

---

## 11. Current project scope

### Implemented now

- English and Hindi dataset ingestion;
- manifest generation;
- transcript cleaning;
- sample-based audio standardization;
- raw waveform and Whisper-feature dataset loaders;
- padded batching;
- small-scale English Whisper fine-tuning;
- training-loss logging;
- validation after the debug steps;
- checkpoint saving; and
- a separate English-to-Hindi text translation demonstration.

### Not yet connected or implemented

- A single end-to-end audio translation flow, such as Hindi audio -> Hindi transcript -> English translation;
- FastAPI routes or a deployed web service;
- full-dataset training on larger compute;
- systematic hyperparameter tuning;
- persistent validation metrics;
- WER/CER evaluation; and
- a complete automated pytest suite.

---

## 12. Recommended presentation conclusion

> Stardust demonstrates the engineering foundation required for a real audio AI system: reliable data ingestion, quality checks, audio standardization, Whisper-compatible feature preparation, batching, fine-tuning, logging, and checkpointing.
>
> The current sample-based run is a deliberate local validation strategy. Once the pipeline is verified, the same design can scale to full datasets and stronger compute resources. The next product step is to connect speech transcription, text translation, evaluation, and an API into one end-to-end service.
