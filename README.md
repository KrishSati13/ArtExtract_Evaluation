# TASK 1

# ArtExtract - CERN HumanAI GSoC Evaluation

This repository contains the evaluation tasks for the ArtExtract project. 

## Task 1: Multi-Task Classification (WikiArt)
* **Architecture:** ResNet18 (Feature Extractor) + Bidirectional LSTM (Spatial Sequence Analysis) + Multi-Task Linear Heads.
* **Hardware:** Trained natively on Apple Silicon (M4 / MPS).
* **Results:** Successfully converged to a loss of ~5.5. 
* **Outlier Detection:** The model successfully identified mislabeled artworks (e.g., confidently identifying a 'Minimalism' labeled red canvas as 'Color Field Painting').
* **Report:** See `Task1_Evaluation.pdf` for training logs and visual outlier proofs.

# TASK 2

# 🏛️ Neural Art Engine: High-Dimensional Visual Similarity

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Accelerated-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-lightgrey)

## 📌 Project Overview
A local, fault-tolerant computer vision pipeline built to process, analyze, and search cultural heritage data. This prototype was developed to demonstrate scalable architecture for Google Summer of Code (GSoC) 2026.

This engine successfully downloaded, cleaned, and extracted deep mathematical features from **22,842 high-resolution paintings** from the National Gallery of Art.

### 🎥 Live Demo
[Watch the 45-Second Web App Demonstration Here] -> *(https://youtu.be/wFEUIXbP5h4)*

## 🧠 Architecture
1. **Data Pipeline:** Automated threaded downloader with built-in SSL bypassing and corrupt-file sweeping to handle massive museum databases.
2. **Feature Extraction:** A modified `ResNet50` model (classification head removed) running on Apple Silicon (`mps`), compressing images into 2048-dimensional vectors.
3. **Similarity Engine:** `scikit-learn` NearestNeighbors calculating absolute Cosine Distance across the entire matrix in milliseconds.
4. **Interactive UI:** A `Gradio` web application allowing users to upload modern photos and instantly retrieve structurally and stylistically similar classical art.

## 🚀 Local Setup
```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/GSoC_ArtExtract.git](https://github.com/yourusername/GSoC_ArtExtract.git)

# 2. Install dependencies
pip install torch torchvision pandas numpy scikit-learn gradio Pillow tqdm

# 3. Run the Web App
python app.py

