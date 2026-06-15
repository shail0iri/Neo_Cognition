# Neo-Cognition

A practical machine-learning project for real-time cognitive-state estimation using computer vision, temporal modeling, and experiment tracking.

Neo-Cognition is designed to show how an AI system can move from raw data to a working real-time application. It combines eye-state analysis, blink dynamics, temporal features, and model evaluation into one end-to-end pipeline that can be explained clearly in interviews and technical discussions.

---

## 1. Project overview

Neo-Cognition estimates human cognitive states such as:

- alertness
- drowsiness
- attention
- fatigue
- stress and cognitive load

The system uses visual cues from webcam input and time-based behavior to produce estimates in real time. The main focus is not only on model accuracy, but also on building a deployable, structured, and reproducible ML workflow.

This project is especially useful for demonstrating:

- applied machine learning
- computer vision
- real-time inference
- experiment tracking
- modular project design

---

## 2. Why this project stands out

This repository is more than a training notebook. It includes:

- a real-time inference path for webcam analysis
- modular training scripts for different model families
- MLflow experiment tracking for reproducibility
- saved model artifacts and evaluation outputs
- a clean project structure suitable for portfolio presentation

That makes it a strong example of end-to-end ML engineering rather than a one-off experiment.

---

## 3. Core capabilities

The project currently supports the following areas:

- CEW-based eye-state classification
- blink detection and blink-rate estimation
- temporal drowsiness modeling using NTHU-style data
- attention and gaze-related modeling using MPIIGaze-style signals
- multimodal fusion and evaluation pipelines
- MLflow-based experiment comparison

---

## 4. Key technical highlights

The system addresses real-world engineering challenges, not just model training:

- robust blink detection using EAR-based logic
- temporal smoothing to reduce noisy frame-level fluctuations
- hysteresis and state-machine logic for more reliable blink detection
- rolling-window blink-rate estimation
- CPU-optimized real-time processing for webcam use
- structured local logging and analysis outputs

These are the kinds of details that interviewers usually appreciate because they show practical engineering judgment.

---

## 5. Tech stack

- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow / Keras
- PyTorch
- Scikit-learn
- XGBoost
- NumPy / Pandas
- Matplotlib / Seaborn
- MLflow

---

## 6. Project structure

- scripts/ — training, evaluation, and model experiments
- src/ — preprocessing, fusion, and real-time inference logic
- outputs/ — processed outputs, summaries, and analysis artifacts
- models/ — saved model checkpoints and final model files
- reports/ — training history and evaluation reports
- notebooks/ — exploratory analysis and experimentation
- logs/ — runtime session and tracking outputs

---

## 7. Quick start

### Environment setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the training workflow

```bash
python scripts/train_cew_cnn_learning.py
python scripts/train_blink_classifier.py
python scripts/train_nthu_temporal.py
python scripts/train_attention_model_mpii.py
```

### Start MLflow UI

```bash
python -m mlflow ui --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///C:/path/to/neo_cognition_mlflow/mlflow.db \
  --default-artifact-root C:/path/to/neo_cognition_mlflow/mlruns
```

### Run the real-time system

```bash
python src/realtime/realtime_cognition.py
```

---

## 8. Verified project status

The current repository has been set up and verified for end-to-end execution with the following observed results:

- CEW CNN: 80.61% test accuracy, 0.4096 test loss
- Blink classifier: 80.79% best accuracy, AUC 0.6974
- NTHU modeling: Random Forest 82.15%, XGBoost 82.72%
- Attention / MPIIGaze-related modeling: R² 0.9996, RMSE 0.0022

These outcomes indicate that the project is working as a real ML pipeline with reproducible outputs rather than as a static demo.

---

## 9. Why this is a strong interview project

This project demonstrates that I can handle the complete ML lifecycle:

1. define a real problem
2. prepare and process data
3. train and compare models
4. evaluate outcomes honestly
5. track experiments for reproducibility
6. build a practical inference path for real users

It is a strong example of practical ML engineering because it combines:

- model development
- system design
- real-time constraints
- evaluation discipline
- project communication

---

## 10. Notes on datasets and artifacts

Large datasets, trained models, and generated experiment files are intentionally not included in this repository because of size and reproducibility considerations. The codebase is structured so the project can be run locally with the appropriate dataset folders and environment setup.

---

## 11. Future direction

Possible next steps include:

- improving model robustness across lighting conditions
- adding more automated evaluation reports
- integrating richer visualization and dashboards
- expanding the fusion layer for more complete cognitive-state estimation

---

## 12. Author

Shail Giri
