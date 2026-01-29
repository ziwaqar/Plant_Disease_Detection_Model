import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loaders
from model import PlantNet
import os

# 1. setup device and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dir = 'data/train'
val_dir = 'data/val'

# get loaders 
train_loader, val_loader, num_classes = get_loaders(train_dir, val_dir, batch_size=32)

# 2. init model 
model = PlantNet(num_classes=num_classes).to(device)

# 3.  loss & optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward pass 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f'epoch: {epoch} | loss: {running_loss/len(train_loader):.3f} | acc: {acc:.2f}%')

# 4. run training
if __name__ == '__main__':
    print(f"starting training on: {device}")
    for epoch in range(1, 11): # start with 10 epochs
        train_one_epoch(epoch)
    
    # save the model 
    torch.save(model.state_dict(), 'results/plant_model_v1.pth')
    print("model saved to results/")