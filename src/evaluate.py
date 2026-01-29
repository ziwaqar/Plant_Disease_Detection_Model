import torch
from sklearn.metrics import classification_report
from data_loader import get_loaders
from model import PlantNet


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    _, val_loader, num_classes = get_loaders('data/train', 'data/val')

    # load model
    model = PlantNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('results/plant_model_v1.pth'))
    model.eval()

    y_true = []
    y_pred = []

    print(f"running evaluation on {device}...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    target_names = ['Apple_Scab', 'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_Healthy']
    print("\n--- Project Evaluation Metrics ---")
    print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == '__main__':
    run_evaluation()