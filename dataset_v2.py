import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class ROPDataset(Dataset):
    """
    Custom Dataset for ROP Classification.
    Maps:
    - Neo_Normal, RetCam_Normal -> Class 0 (Normal)
    - Neo_ROP, RetCam_ROP       -> Class 1 (ROP)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Define class mapping
        class_folders = {
            "Normal": ["Neo_Normal", "RetCam_Normal"],
            "ROP": ["Neo_ROP", "RetCam_ROP"]
        }

        for label_idx, (label_name, folders) in enumerate(class_folders.items()):
            for folder in folders:
                folder_path = os.path.join(root_dir, folder)
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder {folder_path} not found.")
                    continue
                
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.image_paths.append(os.path.join(folder_path, filename))
                        self.labels.append(label_idx)

        print(f"Total images found: {len(self.image_paths)}")
        print(f"Class Distribution: Normal: {self.labels.count(0)}, ROP: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(root_dir, batch_size=16, img_size=512):
    # Aggressive Augmentation for Training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet Stats
    ])

    # Basic Preprocessing for Validation/Test
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = ROPDataset(root_dir)
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Fix seed for reproducibility
    train_data, val_data = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # Apply specific transforms
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = val_transform

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
