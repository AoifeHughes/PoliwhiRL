# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import sqlite3
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F
from plotting import save_comparison_image
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import time

class TrainingLogger:
    def __init__(self, log_dir='training_logs'):
        # Create log directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize CSV file with headers
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.log_dir / f'training_stats_{timestamp}.csv'
        self.stats = []
        
        # Write headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'time_elapsed'])
        
        self.start_time = time.time()
    
    def log_epoch(self, epoch, train_loss, val_loss, learning_rate):
        time_elapsed = time.time() - self.start_time
        stats = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'time_elapsed': time_elapsed
        }
        self.stats.append(stats)
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, learning_rate, time_elapsed])
    
    def plot_training_curves(self):
        df = pd.DataFrame(self.stats)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot losses
        ax1.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o')
        ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(df['epoch'], df['learning_rate'], label='Learning Rate', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / 'training_curves.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, target):
        h_x = x
        h_target = target
        h1_x = self.slice1(h_x)
        h1_target = self.slice1(h_target)
        h2_x = self.slice2(h1_x)
        h2_target = self.slice2(h1_target)
        loss = F.mse_loss(h1_x, h1_target) + F.mse_loss(h2_x, h2_target)
        return loss


class GameBoyNextFrameDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Get all valid samples (excluding last frames of episodes)
        self.cursor.execute("""
            WITH next_frames AS (
                SELECT a.id, a.episode_id, a.wram, a.action, b.image as next_image
                FROM memory_data a
                JOIN memory_data b ON a.id + 1 = b.id AND a.episode_id = b.episode_id
            )
            SELECT id FROM next_frames
        """)
        self.valid_ids = [row[0] for row in self.cursor.fetchall()]
        self.length = len(self.valid_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        current_id = self.valid_ids[idx]
        
        # Get current frame data and next frame image
        self.cursor.execute("""
            SELECT a.wram, a.action, b.image 
            FROM memory_data a
            JOIN memory_data b ON a.id + 1 = b.id
            WHERE a.id = ?
        """, (current_id,))
        
        ram_view_binary, action_binary, next_image_binary = self.cursor.fetchone()

        # Process ram_view
        ram_view = np.load(io.BytesIO(ram_view_binary))
        ram_view = ram_view.astype(np.float32)
        ram_view = (ram_view - ram_view.mean()) / ram_view.std()
        ram_view = torch.from_numpy(ram_view)

        action_int = int.from_bytes(action_binary, byteorder='big')
        action_tensor = torch.tensor(action_int, dtype=torch.long)

        # Process next image
        next_image = Image.open(io.BytesIO(next_image_binary))
        next_image = np.array(next_image)[:, :, :3].transpose(2, 0, 1).astype(np.float32) / 255.0
        next_image = torch.from_numpy(next_image)

        return ram_view, action_tensor, next_image

    def __del__(self):
        self.conn.close()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class NextFramePredictor(nn.Module):
    def __init__(self, num_actions):
        super(NextFramePredictor, self).__init__()

        # Encoder for WRAM
        self.wram_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, 64)
        
        # Combined latent space processing
        self.latent = nn.Sequential(
            nn.Linear(512 * 1024 + 64, 2048),  # Combining WRAM and action features
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 18 * 20 * 64),
        )

        # Decoder (same as before)
        self.decoder = nn.Sequential(
            ConvBlock(64, 128),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2),
            ConvBlock(64, 32),
            nn.Upsample(scale_factor=2),
            ConvBlock(32, 16),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, wram, action):
        # Process WRAM
        x_wram = wram.unsqueeze(1)  # Shape: (batch_size, 1, 8192)
        x_wram = self.wram_encoder(x_wram)
        x_wram = x_wram.view(x_wram.size(0), -1)  # Flatten

        # Process action
        x_action = self.action_embedding(action)  # Shape: (batch_size, 64)

        # Combine features
        combined = torch.cat([x_wram, x_action], dim=1)
        
        # Process through latent space
        x = self.latent(combined)
        x = x.view(x.size(0), 64, 18, 20)
        
        # Decode
        x = self.decoder(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    logger = TrainingLogger()
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (ram_view, action, next_image) in enumerate(progress_bar):
            ram_view = ram_view.to(device)
            action = action.to(device)
            next_image = next_image.to(device)

            optimizer.zero_grad()
            predicted_next_image = model(ram_view, action)
            loss = criterion(predicted_next_image, next_image)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if i % 25 == 0:
                save_comparison_image(next_image[0], predicted_next_image[0], epoch + 1, i=i)

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ram_view, action, next_image in val_loader:
                ram_view = ram_view.to(device)
                action = action.to(device)
                next_image = next_image.to(device)
                predicted_next_image = model(ram_view, action)
                loss = criterion(predicted_next_image, next_image)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch + 1, avg_train_loss, avg_val_loss, current_lr)
        logger.plot_training_curves()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Next Frame Predictor Model")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the SQLite database file")
    parser.add_argument("--num_actions", type=int, required=True, help="Number of possible actions")
    args = parser.parse_args()

    dataset = GameBoyNextFrameDataset(args.db_path)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = NextFramePredictor(num_actions=args.num_actions)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    criterion = nn.MSELoss()
    perceptual_criterion = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )

    # Test the model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for ram_view, action, next_image in test_loader:
            ram_view = ram_view.to(device)
            action = action.to(device)
            next_image = next_image.to(device)
            predicted_next_image = model(ram_view, action)
            loss = criterion(predicted_next_image, next_image)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    print("Training and evaluation completed.")