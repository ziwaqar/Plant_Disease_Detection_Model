# AgriGuard: Custom CNN for Apple Disease Detection
**Sustainable Development Goal Alignment:** SDG 15 (Life on Land) & SDG 2 (Zero Hunger)

## 1. Abstract
AgriGuard is a research-focused computer vision project designed to automate the diagnosis of apple leaf diseases. Using a custom-built Convolutional Neural Network (CNN) architecture, the project classifies images into four categories: Apple Scab, Black Rot, Cedar Apple Rust, and Healthy. This tool supports sustainable agriculture by enabling early-stage disease identification, reducing chemical waste, and protecting crop yields for small-scale farmers.

## 2. Introduction
In many agricultural regions, expert plant pathologists are inaccessible to smallholder farmers. This project is motivated by the need for feasible, low-cost diagnostic tools that can run on consumer-grade hardware. The primary objective was to demonstrate a "Research Mindset" by building a deep learning model from scratch—without utilizing pre-trained weights—to achieve a minimum accuracy of 75% as required by the MS-AI project guidelines.

## 3. Dataset Description
* **Source:** New Plant Diseases Dataset (Kaggle)
* **Sub-sampling:** 4 high-impact Apple categories (1,943 images for final validation).
* **Preprocessing:** * **Normalization:** Scaled pixel values using ImageNet mean and standard deviation.
    * **Augmentation:** Implemented Random Horizontal Flips and Rotations (10°) to simulate real-world field conditions and prevent overfitting.
    * **Splitting:** Stratified 80/10/10 split (Train/Val/Test) to ensure unbiased evaluation.

## 4. Tech Stack
* **Language:** Python 3.10+
* **Framework:** PyTorch / Torchvision
* **Environment:** Local Workstation (CUDA 12.x)
* **Tools:** Git/GitHub, Matplotlib (Visualizations), Scikit-Learn (Metrics)

## 5. Methodology
### Architecture
Instead of using a production-grade ResNet, a **Custom CNN** was architected to focus on forward/backward pass understanding:
* **Conv Layer 1:** 32 filters (3x3), ReLU activation, Max-Pooling (2x2).
* **Conv Layer 2:** 64 filters (3x3), ReLU activation, Max-Pooling (2x2).
* **Conv Layer 3:** 128 filters (3x3), ReLU activation, Max-Pooling (2x2).
* **Regularization:** Dropout (0.5) to ensure the research model generalizes well to unseen data.
* **Fully Connected:** Dense layers transitioning from flattened features to the 4-class Softmax output.

## 6. Experimental Setup
* **Hardware:** NVIDIA RTX 5070.
* **Software:** Python 3.10, PyTorch, Scikit-Learn, Matplotlib.
* **Hyperparameters:** Adam Optimizer, Learning Rate 0.001, Batch Size 32, 10 Epochs.

## 6. Results & Performance
The model achieved significantly higher results than the required 75% baseline.

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Apple Scab | 0.98 | 0.98 | 0.98 |
| Apple Black Rot | 0.98 | 0.99 | 0.99 |
| Apple Cedar Rust | 0.99 | 0.99 | 0.99 |
| Apple Healthy | 0.99 | 0.99 | 0.99 |
| **Accuracy** | | | **0.99** |



## 7. Discussion
* **What Worked:** The custom 3-layer depth was sufficient to capture the "Rusty" lesions of Cedar Rust, leading to 100% confidence on several test images.
* **Challenges:** Initial issues with Windows Multiprocessing were resolved by implementing proper `__main__` entry points.
* **Lessons Learned:** High accuracy in controlled datasets (Softmax Saturation) does not always equate to real-world robustness; hence, the use of unseen test sets is vital for research integrity.

## 8. Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. To reproduce results: `python src/evaluate.py`.
4. To test a sample: `python src/predict.py`.

## 9. Conclusion & Future Work
This project successfully proves the feasibility of using custom CNNs for agricultural monitoring. Future improvements include implementing **Vision Transformers (ViT)** to compare spatial attention against standard convolutions.

