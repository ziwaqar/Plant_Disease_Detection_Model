# Custom CNN for Plant Disease Detection
**Sustainable Development Goal Alignment:** SDG 15 (Life on Land) & SDG 2 (Zero Hunger)

## Project Overview
This research-focused project develops a custom Convolutional Neural Network (CNN) from scratch to identify diseases in agricultural crops. By providing an automated diagnostic tool, this project aims to support small-scale farmers in reducing crop loss and promoting sustainable land use.

## Research Objectives
* Implement a custom neural network architecture without using pre-trained weights.
* Achieve a minimum classification accuracy of 75% on the target dataset.
* Evaluate performance using Research-standard metrics: Precision, Recall, and F1-Score.

## Dataset Information
* **Source:** New Plant Diseases Dataset (Kaggle)
* **Size:** ~500+ images (focused subset)
* **Classes:** [e.g., Tomato Healthy, Tomato Late Blight, Potato Early Blight]
* **Preprocessing:** 128x128 resizing, Normalization, and Domain-specific Augmentation (Flips/Rotations).

## Methodology 
Unlike production-grade systems that use Transfer Learning, this project utilizes a **Custom CNN architecture** to demonstrate a deep understanding of forward/backward passes and feature extraction layers.

## Tech Stack
* **Language:** Python 3.10+
* **Framework:** PyTorch / Torchvision
* **Environment:** Local Workstation (NVIDIA RTX 5080, CUDA 12.x)
* **Tools:** Git/GitHub, Matplotlib (Visualizations), Scikit-Learn (Metrics)

