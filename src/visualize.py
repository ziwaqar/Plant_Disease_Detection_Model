import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import label_binarize # Fix is here
from data_loader import get_loaders
from model import PlantNet

def generate_research_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, num_classes = get_loaders('data/train', 'data/val')
    
    # load model
    model = PlantNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('results/plant_model_v1.pth'))
    model.eval()

    y_true, y_pred, y_scores = [], [], []
    classes = ['Apple_Scab', 'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_Healthy']

    print("Collecting data for visualization...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # --- GRAPH 1: CONFUSION MATRIX ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Disease Classification Heatmap')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.savefig('results/vis_confusion_matrix.png')
    plt.close() # Closes the plot so it doesn't pop up in the notebook automatically

    # --- GRAPH 2: PRECISION-RECALL CURVE (Research Requirement) ---
    plt.figure(figsize=(8, 6))
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class: {classes[i]}')
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision-Recall Curve (Multi-class)")
    plt.savefig('results/vis_precision_recall.png')
    plt.close()

    # --- GRAPH 3: CLASS ACCURACY COMPARISON ---
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_acc = cm_normalized.diagonal() * 100
    sns.barplot(x=classes, y=class_acc, palette='viridis')
    plt.title('Individual Class Accuracy (%)')
    plt.ylabel('Accuracy Percentage')
    plt.ylim(0, 110)
    plt.savefig('results/vis_class_accuracy.png')
    plt.close()

if __name__ == "__main__":
    generate_research_plots()