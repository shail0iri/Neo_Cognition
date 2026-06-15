# Neo-Cognition

A practical machine-learning project focused on real-time cognitive state estimation using computer vision, temporal modeling, and experiment tracking.

This repository brings together multiple components of an applied ML system: data preparation, model training, evaluation, logging, and a real-time inference path. The goal is not just to train models, but to build a project that behaves like a real engineering workflow rather than a single notebook experiment.

## What this project is

Neo-Cognition is a multimodal AI project for estimating human cognitive states such as alertness, drowsiness, attention, and fatigue from visual cues and time-based behavior. The core idea is to combine:

- eye-state analysis,
- blink dynamics,
- temporal features,
- and lightweight model evaluation in a reproducible pipeline.

It is designed to be useful both as a research project and as a portfolio project for interviews and technical discussions.

## Why this repository stands out

This project is not only about model training. It includes:

- a real-time inference path for webcam-based analysis,
- modular training scripts for different model families,
- MLflow experiment tracking for reproducibility,
- evaluation outputs and saved artifacts,
- and a structured project layout that is easier to explain in interviews.

That makes it a good example of end-to-end ML engineering rather than a one-off experiment.

## Current project scope

The repository currently supports:

- CEW-based eye-state classification
- blink detection and model comparison
- NTHU temporal drowsiness modeling
- attention / gaze-related modeling using MPIIGaze-style data
- MLflow-based experiment tracking and run comparison

## Quick start

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the training scripts

```bash
python scripts/train_cew_cnn_learning.py
python scripts/train_blink_classifier.py
python scripts/train_nthu_temporal.py
python scripts/train_attention_model_mpii.py
```

### 3. Start MLflow UI

```bash
python -m mlflow ui --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///C:/path/to/neo_cognition_mlflow/mlflow.db \
  --default-artifact-root C:/path/to/neo_cognition_mlflow/mlruns
```

### 4. Run the live system

```bash
python src/realtime/realtime_cognition.py
```

## What I have implemented

- Training pipelines for multiple model types
- MLflow logging for parameters, metrics, and artifacts
- CEW data-path handling for stable local execution
- Structured outputs for evaluation and analysis
- A real-time inference flow for webcam-based cognitive-state estimation

## Verified results (latest run)

The following results were verified from the current project setup:

- CEW CNN: 80.61% test accuracy, 0.4096 test loss
- Blink classifier: 80.79% best accuracy, AUC 0.6974
- NTHU model: Random Forest 82.15%, XGBoost 82.72%
- Attention / MPIIGAZE model: R² 0.9996, RMSE 0.0022

These numbers show that the project is functioning end to end, and the experiments are being tracked in MLflow rather than being left as ad-hoc outputs.

## Project structure

- scripts/ — training and evaluation scripts
- src/ — preprocessing, fusion, and realtime inference logic
- outputs/ — generated analysis and processed summaries
- models/ — saved model checkpoints and final model files
- reports/ — training history and evaluation artifacts
- notebooks/ — exploratory analysis and experiments

## Notes on datasets and artifacts

Large datasets, generated model files, and experiment artifacts are intentionally not included in this repository for size and reproducibility reasons. The code and pipeline are structured so the project can be run locally with the appropriate data folder setup.

## Tech stack

- Python
- TensorFlow / Keras
- PyTorch
- Scikit-learn
- XGBoost
- OpenCV
- Pandas / NumPy
- Matplotlib / Seaborn
- MLflow

## Why this is useful for an interview

This project demonstrates that I can work across the full ML lifecycle:

- understand the problem,
- build and train models,
- evaluate results honestly,
- log experiments for comparison,
- and structure the project for real-world use.

It is a good example of a practical ML engineering project rather than a generic tutorial repository.

## Author

Shail Giri

ML Engineer | Computer Vision | Multimodal AI | Real-Time Systems
