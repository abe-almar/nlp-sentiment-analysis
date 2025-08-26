# 🐦 Tweet Sentiment Analyzer – NLP with FastAPI

![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688)
![Model](https://img.shields.io/badge/Model-Keras%20CNN-blueviolet)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)


## 1. Overview

This is a lightweight FastAPI application for entity-level sentiment analysis of short social media text (e.g., tweets). Given a `text` and a target `entity`, the service predicts one of: Negative, Neutral, Positive, and returns a confidence score.

Expected input format:
- **text**: a short piece of text (e.g., a tweet) that may contain one or more entities
- **entity**: the specific entity within the text to evaluate sentiment for
- **Output**: the model’s predicted class label.

Examples:
- **Single-entity text**
  - Text: "I love my new Tesla Model 3, it's amazing!"
  - Entity: "Tesla"
  - Output: Positive

## 2. Project Organization

```
├── LICENCE                      <- Project license
├── README.md                    <- The top-level README for developers using this project
├── Dockerfile                   <- Container image to run the API consistently
├── Makefile                     <- Automation for formatting and linting
├── pyproject.toml               <- Project metadata and dependencies
├── uv.lock                      <- Dependency lockfile (uv)
│
├── docs/                        <- Additional docs
│   └── API.md                   <- API endpoints and examples
│
├── notebooks/                   <- Jupyter notebooks (experiments, demos)
│   ├── module-testing.ipynb
│   └── nlp-sentiment-analysis.ipynb
│
├── data/                        <- Optional: local data directory (excluded if private)
│
├── models/                      <- Alternative/trained models (legacy artifacts)
│
└── src/                         <- Python source code
    ├── __init__.py
    ├── formatting.py            <- Input formatting/preprocessing
    ├── train.py                 <- Model training and saving
    ├── main.py                  <- FastAPI app
    ├── models/                  <- Runtime model used by the API
    │   └── full_model_1D.keras
    └── utils/
        └── logger.py            <- Logging utility
```

## 3. Requirements

This project requires a `.env` file to function correctly. The `.env` file stores important environment variables, such as the path to your data folder. **You must have this file configured for the code to run.**

A template file named `.env.example` is provided for your convenience. To set up your environment variables:

1. Duplicate the `.env.example` file and rename the copy to `.env`:
- In Linux/macOS:
```bash
cp .env.example .env
```
- In Windows:
```bash
copy .env.example .env
```

2. Open the newly created `.env` file and add or update your environment variables as needed. For example:

```bash
# Required
DATA_FOLDER="/absolute/path/to/dataset/folder"  # folder containing twitter_training.csv and twitter_validation.csv
# Optional (CUDA environment hint for some setups)
XLA_FLAGS=--xla_gpu_cuda_data_dir="/usr/lib/cuda"
```

⚠️ **Never commit your `.env` file to the repository!** Only `.env.example` should be tracked in version control as a template for others.

It's important to write the path to the data folder without any commas (",") or you can run into issues running the code from the command prompt (some characters can be interpreted as ASCII bell character)

## 4. Running the project

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

### Create and activate the virtual environment

```bash
make env
```

### Code formatting

```bash
make format   # auto-format (ruff/black/isort via uv)
```

### Code linting

```bash
make lint     # lint checks (ruff)
```

### Download datasets and set DATA_FOLDER in .env

```bash
make dataset
```
**Make sure you have already created your .env file before running this command**

### Train the model

Ensure `.env` has `DATA_FOLDER` set, then run:

```bash
make train
```

This will train and save the end-to-end model to `/models/full_model_1D.keras`.

## Documentation

- [FastAPI application](./docs/API.md)