# AgriGuard Project  
**Automated Plant Disease Detection System**

**Version:** 1.0.0  
**Environment:** Python / PyTorch / CUDA

---

## 1. System Overview

**AgriGuard** is a modular software pipeline designed for the classification of apple leaf diseases. The system is divided into four main functional blocks:

- **Data Ingestion**
- **Model Architecture**
- **Training Engine**
- **Inference & Visualization**

The architecture emphasizes scalability, reproducibility, and research-grade experimentation.

---

## 2. Directory Structure

A modular approach was adopted to ensure maintainability and separation of concerns.

```plaintext
Plant_Disease_Detection/
├── data/               # Persistent storage for raw and processed imagery
├── notebooks/          # Exploratory Data Analysis (EDA) and prototyping
├── results/            # Model checkpoints (.pth) and generated visuals
├── src/                # Source code repository
│   ├── data_loader.py  # Data pipeline and augmentation logic
│   ├── model.py        # Neural Network architecture definition
│   ├── train.py        # Training loop and optimization logic
│   ├── evaluate.py     # Metric calculation (F1, Precision, Recall)
│   ├── predict.py      # Real-time single-image inference
│   └── visualize.py    # Research graph generation
├── requirements.txt    # Dependency manifest
└── README.md           # Project entry point
```

---

## 3. Technical Specification

### 3.1 Hardware Environment

| Component   | Specification |
|------------|---------------|
| Processor  | High-performance multi-core CPU |
| Accelerator| NVIDIA GeForce RTX 5070 |
| Driver     | CUDA 11.x / 12.x compatible |

---

### 3.2 Software Dependencies

| Category        | Tools |
|-----------------|-------|
| Language        | Python 3.10+ |
| Deep Learning   | PyTorch 2.x, Torchvision |
| Data Processing | NumPy, Pandas |
| Visualization   | Matplotlib, Seaborn |
| Metrics         | Scikit-Learn |

---

## 4. Data Engineering

### 4.1 Data Pipeline (`data_loader.py`)

The pipeline uses a **Subset filtering strategy** to extract a focused 4-class Apple pathology dataset from a larger multi-class corpus.

#### Normalization

$$
\text{Input} = \frac{(\text{Pixel} - \mu)}{\sigma}
$$

Where:

- $\mu = [0.485, 0.456, 0.406]$
- $\sigma = [0.229, 0.224, 0.225]$

#### Augmentation

- Random horizontal flipping  
- Random affine rotations  

These augmentations improve generalization and prevent orientation memorization.

---

### 4.2 Class Mapping

| Index | Class Name   | Description |
|------:|-------------|-------------|
| 0 | Apple_Scab | *Venturia inaequalis* fungus |
| 1 | Black_Rot | *Diplodia seriata* fungus |
| 2 | Cedar_Rust | *Gymnosporangium juniperi-virginianae* |
| 3 | Healthy | Baseline control group |

---

## 5. Model Architecture (`model.py`)

The system utilizes **PlantNet**, a custom 3-stage Convolutional Neural Network.
## Model Architecture Diagram

![PlantNet Model Architecture](assets/plantnet_architecture.png)

| Layer Type | Configuration | Output Shape | Purpose |
|-----------|---------------|--------------|---------|
| Input | RGB Image | (3, 128, 128) | Raw data |
| Conv2D + ReLU | 32 filters, 3×3 | (32, 126, 126) | Low-level edges |
| Max Pool | 2×2 kernel | (32, 63, 63) | Spatial reduction |
| Conv2D + ReLU | 64 filters, 3×3 | (64, 61, 61) | Texture patterns |
| Max Pool | 2×2 kernel | (64, 30, 30) | Feature selection |
| Conv2D + ReLU | 128 filters, 3×3 | (128, 28, 28) | Disease lesions |
| Max Pool | 2×2 kernel | (128, 14, 14) | Complexity reduction |
| Dropout | p = 0.5 | (128, 14, 14) | Overfitting prevention |
| Fully Connected | 128×14×14 → 4 | (4) | Classification logits |

---

## 6. Training & Optimization Logic

### 6.1 Loss Function

**Cross-Entropy Loss** is used to measure the divergence between predicted probabilities and ground-truth labels.

---

### 6.2 Optimization Algorithm

The **Adam Optimizer** is implemented with a learning rate:

$$
\alpha = 0.001
$$

Adam provides adaptive learning rates, accelerating convergence in sparse-gradient agricultural imagery.

---

### 6.3 Backpropagation

Each training iteration performs:

1. Forward pass
2. `loss.backward()` to compute gradients
3. `optimizer.step()` to update weights

---

## 7. Evaluation & Inference

### 7.1 Metric Suite (`evaluate.py`)

- **Precision** – Correctness of positive predictions  
- **Recall** – Ability to detect all positive instances  
- **F1-Score** – Harmonic mean of Precision and Recall

---

### 7.2 Inference Engine (`predict.py`)

The inference module:

1. Loads trained `.pth` weights  
2. Applies preprocessing and normalization  
3. Performs forward inference  
4. Applies Softmax to output confidence percentages

---

## 8. Installation & Deployment

### 8.1 Environment Setup

```bash
conda create -n plant_project python=3.10
conda activate plant_project
pip install torch torchvision seaborn scikit-learn matplotlib
```

---

### 8.2 Execution Commands

```bash
# Train the model
python src/train.py

# Evaluate the model
python src/evaluate.py

# Run inference
python src/predict.py --image path/to/leaf.jpg
```

---

## 9. Future Scalability

- **API Integration**: Convert inference logic into a FastAPI service  
- **Model Compression**: Quantize weights to 16-bit precision for mobile deployment  
- **Extended Dataset**: Add Grape and Tomato disease classes to broaden SDG impact

---

**End of Technical Documentation**