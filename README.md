# 🎙️ AI Audio Translation System (End-to-End ML Pipeline)

## 📌 Project Overview

This project builds a **production-ready AI pipeline** that can:

* Convert **audio → text (speech-to-text)**
* Support **multiple languages (English, Hindi)**
* Handle **large-scale datasets**
* Prepare data for **real-time translation systems**
* Follow **enterprise-grade architecture**

The goal is to simulate a **banking-grade AI system** used in:

* Call center automation
* Multilingual customer support
* Voice-based assistants

---

## 🧠 Why This Project Matters

In real-world systems (e.g., banks, fintech companies), AI pipelines must:

* Handle **large, messy datasets**
* Be **scalable and modular**
* Support **multiple languages**
* Ensure **data quality and reproducibility**
* Be **easy to debug and extend**

This project focuses on building that **foundation correctly**.

---

## 🏗️ Project Architecture

```
Raw Data → Inventory → Manifest → Clean Manifest → Standardized Audio → Dataset Loader → Training
```

Each step is **modular and isolated**, which is how real ML systems are built.

---

## 📁 Folder Structure

```
data/
├── raw/                # Original downloaded datasets (never modified)
├── interim/            # Extracted datasets
├── metadata/           # Inventory + manifest files
├── processed/          # Cleaned and structured data
├── standardized/       # Converted audio (WAV, 16kHz)

src/
├── ingestion/          # Data discovery and loading
├── preprocessing/      # Data cleaning and transformation
├── training/
│   ├── datasets/       # PyTorch dataset classes
│   ├── utils/          # Collate functions and helpers
│   └── test files      # Debug/testing scripts
```

---

## 🔄 Pipeline Breakdown (Step-by-Step)

---

### 1️⃣ Data Collection

**What we did:**

* Used **Mozilla Common Voice datasets (English + Hindi)**

**Why:**

* Free and legal
* Multilingual
* Real-world speech data

**Alternative:**

* Google Speech datasets (paid)
* Custom recorded data

---

### 2️⃣ Data Inventory (`data_loader.py`)

**What we did:**

* Scanned all folders recursively
* Created a CSV of all audio files

**Why:**

* Datasets had complex nested structures
* Needed a **single source of truth**

**Output:**

```
language, file_name, file_path, extension
```

**Alternative:**

* Direct file loading during training (❌ not scalable)

---

### 3️⃣ Training Manifest (`training_manifest.py`)

**What we did:**

* Matched audio files with transcripts
* Supported **multiple dataset formats**

**Problem solved:**

* English dataset used `filename, text`
* Hindi dataset used `path, sentence`

**Why:**

* Real-world datasets are inconsistent

**Output:**

```
language, split, audio_file, audio_path, transcript
```

**Alternative:**

* Hardcoding schema (❌ breaks for different datasets)

---

### 4️⃣ Clean Manifest (`preprocess_manifest.py`)

**What we did:**

* Removed:

  * missing audio
  * empty transcripts
  * duplicates
* Normalized text
* Added quality metrics

**Why:**

* Clean data = better model performance
* Prevent runtime errors

**Output:**

```
manifest_clean.csv
```

**Alternative:**

* Skip cleaning (❌ leads to model failure)

---

### 5️⃣ Audio Standardization (`audio_standardizer.py`)

**What we did:**

* Converted all audio to:

  * WAV format
  * 16kHz sample rate
  * mono channel

**Why:**

* Models require consistent input format
* Reduces training errors

**Output:**

```
data/standardized/<language>/<split>/*.wav
```

**Alternative:**

* Train on raw formats (❌ unstable training)

---

### 6️⃣ PyTorch Dataset (`audio_dataset.py`)

**What we did:**

* Built a dataset class to:

  * load audio
  * return waveform + transcript

**Why:**

* PyTorch requires structured dataset objects

**Output:**

```
{
  waveform,
  sample_rate,
  transcript,
  language,
  split
}
```

**Alternative:**

* Load inside training loop (❌ messy code)

---

### 7️⃣ Custom Collate Function (`collate_fn.py`)

**Problem:**
Audio files have different lengths → batching fails

**What we did:**

* Implemented padding for variable-length audio

**Why:**

* Required for batching in PyTorch

**Output:**

```
padded_waveforms, waveform_lengths, transcripts
```

**Alternative:**

* Trim audio (❌ lose information)
* Fixed-length input (❌ inefficient)

---

## ⚠️ Key Challenges Solved

### ✔ Dataset inconsistency

Different formats handled dynamically

### ✔ Nested folder structures

Recursive file discovery

### ✔ Audio format differences

Standardization pipeline

### ✔ Variable-length batching

Custom collate function

### ✔ Encoding issues (Hindi text)

UTF-8 handling

---

## 🚀 Current Status

✅ Data ingestion pipeline
✅ Manifest generation
✅ Data cleaning
✅ Audio standardization
✅ Dataset loader
✅ Batch handling

---

## 🔜 Next Steps

* Load **Hugging Face processor (Whisper)**
* Convert:

  * waveform → features
  * text → tokens
* Build training loop
* Add translation layer
* Deploy API (FastAPI + Azure)

---

## 🏦 Real-World Applications

This system can be used in:

* Banking call centers
* Customer support AI
* Voice assistants
* Real-time translation tools

Example:

```
Hindi speech → Text → English translation → Agent response
```

---

## 🧪 How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Run dataset test

```
python -m src.training.test_dataset_loader
```

---

## 💡 Why This Project Stands Out

This is not just a model.

It demonstrates:

* Data engineering
* ML pipeline design
* PyTorch fundamentals
* Real-world problem solving
* Scalable architecture

---

## 👨‍💻 Author

Built as a learning + production-style AI system
for real-world deployment readiness.

---
