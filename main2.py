import torch
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

class GeoPredictor(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.regressor = torch.nn.Linear(clip_model.config.hidden_size, 2)

    def forward(self, pixel_values):
        outputs = self.clip_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    criterion = torch.nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        for images, coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, coords = images.to(device), coords.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, coords in val_loader:
                images, coords = images.to(device), coords.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, coords).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader)}")

def evaluate_model(model, test_loader, device):
    model.eval()
    distances = []
    
    with torch.no_grad():
        for images, coords in tqdm(test_loader, desc="Evaluating"):
            images, coords = images.to(device), coords.to(device)
            predicted_coords = model(images)
            
            for true, pred in zip(coords.cpu().numpy(), predicted_coords.cpu().numpy()):
                distance = geodesic(true, pred).kilometers
                distances.append(distance)
    
    avg_distance = np.mean(distances)
    print(f"Average distance between predicted and actual coordinates: {avg_distance:.2f} km")

def main():
    # Hyperparameters
    batch_size = 32
    grad_accum_steps = 8
    learning_rate = 1e-6
    weight_decay = 1e-3
    warmup_epochs = 0.2
    num_epochs = 3
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GeoImageDataset('datalink.csv', 'path/to/image/directory', transform=transform)
    train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
    model = GeoPredictor(clip_model).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))

    # Training
    train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device)

    # Save model
    torch.save(model.state_dict(), 'geo_predictor_model.pth')

    # Evaluation
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
