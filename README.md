# ArtExtract - CERN HumanAI GSoC Evaluation

This repository contains the evaluation tasks for the ArtExtract project. 

## Task 1: Multi-Task Classification (WikiArt)
* **Architecture:** ResNet18 (Feature Extractor) + Bidirectional LSTM (Spatial Sequence Analysis) + Multi-Task Linear Heads.
* **Hardware:** Trained natively on Apple Silicon (M4 / MPS).
* **Results:** Successfully converged to a loss of ~5.5. 
* **Outlier Detection:** The model successfully identified mislabeled artworks (e.g., confidently identifying a 'Minimalism' labeled red canvas as 'Color Field Painting').
* **Report:** See `Task1_Evaluation.pdf` for training logs and visual outlier proofs.

