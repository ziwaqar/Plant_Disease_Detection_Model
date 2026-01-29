import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_loaders(train_dir, val_dir, batch_size=32):
    # Only use these 4 specific classes to save time
    target_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])

    # Load full datasets
    full_train_ds = datasets.ImageFolder(train_dir, transform=transform)
    full_val_ds = datasets.ImageFolder(val_dir, transform=transform)

    # Filter indices for target classes
    def get_indices(dataset):
        indices = []
        # find the numeric index for our chosen class names
        target_idx = [dataset.class_to_idx[c] for c in target_classes]
        for i, (_, label) in enumerate(dataset.samples):
            if label in target_idx:
                indices.append(i)
        return indices

    train_subset = Subset(full_train_ds, get_indices(full_train_ds))
    val_subset = Subset(full_val_ds, get_indices(full_val_ds))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, len(target_classes)