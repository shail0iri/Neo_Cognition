# 🧠 Neo-Cognition  
### Multimodal Real-Time Cognitive State Estimation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Multimodal](https://img.shields.io/badge/multimodal-AI-orange.svg)

Neo-Cognition is an end-to-end **multimodal AI system** that estimates human cognitive states such as **alertness, drowsiness, attention, and fatigue** in real time using **visual cues and temporal dynamics**.

The project is designed as a **full ML engineering pipeline**, covering:
data preprocessing → feature extraction → model training → multimodal fusion → real-time inference.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/shail0iri/Neo_Cognition.git
cd Neo_Cognition

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run real-time demo
python src/realtime/webcam_fusion.py

```

 Features

 Real-time webcam inference using OpenCV + MediaPipe

 Eye-state classification (CNN) trained on CEW (Closed Eyes in the Wild)

 Temporal drowsiness modeling using NTHU Drowsy Driver Dataset

 Attention estimation from gaze and blink dynamics (MPIIGaze)

 Multimodal fusion engine combining CNN outputs and temporal ML models

 Evaluation & visualization (confusion matrices, training curves)

 Modular architecture with clean separation of preprocessing, training, fusion, and inference


System Architecture 

                 Webcam
                   │
                   ▼
          MediaPipe Face Mesh
                   │
          ┌────────┴────────┐
          ▼                 ▼
     Eye ROI           Temporal Features
   (CEW CNN)     (Blink, EAR, Gaze)
          │                 │
          └────────┬────────┘
                   ▼
          Multimodal Fusion Engine
                   │
                   ▼
        Cognitive State Estimation
     (Alertness • Drowsiness • Attention)

Project Structure
neo_cognition/
├── src/
│   ├── preprocess/     # Dataset-specific preprocessing pipelines
│   ├── fusion/         # Fusion dataset creation & fusion models
│   └── realtime/       # Real-time inference & dashboard
│
├── scripts/            # Training & testing scripts
├── notebooks/          # EDA and experimentation
│
├── data/               # Raw datasets (ignored on GitHub)
├── outputs/            # Generated features & artifacts (ignored)
├── models/             # Trained model checkpoints (ignored)
├── reports/            # Metrics & plots (ignored)
│
├── requirements.txt
├── .gitignore
└── README.md
⚠️ Large artifacts (data/, outputs/, models/, reports/) are intentionally excluded from version control.

Datasets Used

This project integrates multiple public datasets:

Dataset	Purpose
CEW	Eye-state classification
NTHU DDD	Temporal drowsiness detection
MPIIGaze	Gaze & attention estimation
Eyeblink8	Blink dynamics
CLAS	Cognitive load annotations

⚠️ Due to size and licensing restrictions, datasets are not included in this repository.

Installation Requirements

Python 3.8+

Webcam (for real-time inference)

8GB+ RAM recommended

Usage
1️⃣ Train individual models
python scripts/train_blink_classifier.py
python scripts/train_cew_cnn_learning.py
python scripts/train_nthu_temporal.py
python scripts/train_attention_model_mpii.py

2️⃣ Test multimodal fusion
python scripts/test_fusion_cew_nthu.py

3️⃣ Run real-time system
python src/realtime/webcam_fusion.py

Results & Evaluation

Eye-state CNN shows robust performance across varying lighting conditions

Temporal models improve drowsiness detection over frame-level approaches

Multimodal fusion produces smoother and more reliable cognitive estimates

Real-time inference runs efficiently on CPU-based systems

Evaluation artifacts (confusion matrices, training curves) are generated locally.

🧩 Why This Project Matters

Most ML projects stop at single-model training.

Neo-Cognition focuses on:

System-level ML engineering

Multimodal reasoning

Temporal modeling

Real-time deployment

Clean, maintainable code structure

This makes it closer to production-style cognitive AI systems used in
automotive safety, HCI, and applied AI research.

🛠️ Tech Stack

Computer Vision: OpenCV, MediaPipe

Deep Learning: TensorFlow / Keras, PyTorch

Machine Learning: Scikit-learn, XGBoost

Data Processing: NumPy, Pandas, Matplotlib

Real-Time UI: Streamlit

Development: Python, Git

🤝 Contributing

Contributions are welcome.

Fork the repository

Create a feature branch

Commit your changes

Open a Pull Request

👤 Author

Shail Giri
AI / ML Engineer — Computer Vision • Multimodal Systems • Real-Time AI

GitHub: https://github.com/shail0iri

⭐ If you find this project useful, please consider starring the repository.


---

## ✅ FINAL STEP (DON’T SKIP)

After pasting:

```bash
git add README.md
git commit -m "Add cleaned and accurate project README"
git push
