import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import PlantNet
import os

def run_prediction():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    classes = ['Apple_Scab', 'Apple_Black_rot', 'Apple_Cedar_rust', 'Apple_Healthy']
    
    # 2. Load Model
    model = PlantNet(num_classes=len(classes)).to(device)
    model_path = 'results/plant_model_v1.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")

    # 3. Preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4.image to test 
   
    test_base_path = 'data/test'
    test_img = "AppleScab1.JPG"
    
    for root, dirs, files in os.walk(test_base_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_img = os.path.join(root, file)
                break
        if test_img: break

    if test_img is None:
        print(f"Error: No images found in {test_base_path}")
        return

    print(f"Testing on image: {test_img}")

    # 5. Inference
    img = Image.open(test_img).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        
    print("-" * 30)
    print(f"RESULT: {classes[pred.item()]}")
    print(f"CONFIDENCE: {conf.item()*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    run_prediction()