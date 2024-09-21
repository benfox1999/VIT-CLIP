import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from geopy.distance import geodesic

class GeoImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None, names=['latitude', 'longitude', 'id'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data) * 4  # 4 images per location

    def __getitem__(self, idx):
        row = idx // 4
        img_idx = idx % 4
        lat = self.data.iloc[row, 0]
        lon = self.data.iloc[row, 1]
        img_id = self.data.iloc[row, 2]
        
        img_path = os.path.join(self.img_dir, f"{img_id}${img_idx}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([lat, lon], dtype=torch.float)

class GeoPredictor(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.regressor = nn.Linear(clip_model.config.hidden_size, 2)
        print("GeoPredictor model initialized")

    def forward(self, pixel_values):
        outputs = self.clip_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    print("Starting model training")
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, coords) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, coords = images.to(device), coords.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, coords in val_loader:
                images, coords = images.to(device), coords.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, coords).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
    
    print("Model training completed")

def evaluate_model(model, test_loader, device):
    print("Starting model evaluation")
    model.eval()
    distances = []
    
    with torch.no_grad():
        for i, (images, coords) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, coords = images.to(device), coords.to(device)
            predicted_coords = model(images)
            
            for true, pred in zip(coords.cpu().numpy(), predicted_coords.cpu().numpy()):
                print(true, pred)
                distance = geodesic(true, pred).kilometers
                distances.append(distance)
            
            if i % 50 == 0:
                print(f"Batch {i}, Current average distance: {np.mean(distances):.2f} km")
    
    avg_distance = np.mean(distances)
    print(f"Evaluation completed. Average distance between predicted and actual coordinates: {avg_distance:.2f} km")

def main():
    print("Starting GeoPredictor main function")
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-6
    weight_decay = 1e-3
    num_epochs = 3
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    print(f"Hyperparameters: batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}, num_epochs={num_epochs}")

    # Data preparation
    print("Preparing dataset and dataloaders")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GeoImageDataset('./datalink.csv', './dataset/dataset', transform=transform)
    print(f"Total dataset size: {len(dataset)}")
    
    train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True)
    print("Dataloaders created")

    # Model setup
    print("Setting up model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    clip_model = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    print("CLIP model loaded")
    
    model = GeoPredictor(clip_model)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU or CPU")
    
    model = model.to(device)
    print("Model moved to device")

    # Optimizer and scheduler
    print("Setting up optimizer and scheduler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))
    print("Optimizer and scheduler created")

    # Training
    print("Starting training process")
    train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device)

    # Save model
    print("Saving trained model")
    torch.save(model.state_dict(), 'geo_predictor_model.pth')
    print("Model saved as 'geo_predictor_model.pth'")

    # Evaluation
    print("Starting evaluation process")
    evaluate_model(model, test_loader, device)

    print("GeoPredictor main function completed")

if __name__ == "__main__":
    main()
