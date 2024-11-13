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


class GameBoyDataset(Dataset):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM memory_data")
        self.length = self.cursor.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.cursor.execute(
            "SELECT wram, image FROM memory_data WHERE id=?", (idx + 1,)
        )
        ram_view_binary, image_binary = self.cursor.fetchone()

        # Process ram_view
        ram_view = np.load(io.BytesIO(ram_view_binary))
        ram_view = ram_view.astype(np.float32)
        ram_view = (ram_view - ram_view.mean()) / ram_view.std()  # Standardization
        ram_view = torch.from_numpy(ram_view)

        # Process image
        image = Image.open(io.BytesIO(image_binary))
        image = np.array(image)[:, :, :3].transpose(2, 0, 1).astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        return ram_view, image

    def __del__(self):
        self.conn.close()


class PixelwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ImprovedGBCModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Increased initial channels and added normalization
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        # Gradual reduction in dense layer sizes with dropout
        self.dense = nn.Sequential(
            nn.Linear(128 * 4096, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 18 * 20 * 128),
        )

        # Increased channels in decoder
        self.initial_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        # Multiple residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128)
        )

        self.color_processor = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        self.upsampler = nn.Upsample(
            size=(144, 160), mode="bilinear", align_corners=True
        )

    def forward(self, x):
        # Add noise during training for regularization
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = x.view(x.size(0), 128, 18, 20)

        x = self.initial_conv(x)
        x = self.residual_blocks(x)
        x = self.upsampler(x)
        x = self.color_processor(x)
        x = self.final(x)

        return torch.sigmoid(x)  # Remove quantization during training


class GBCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Basic reconstruction loss
        reconstruction_loss = F.mse_loss(pred, target)

        # Edge preservation loss using gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        edge_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        # Color quantization loss
        quantized_pred = (pred * 31).round() / 31
        quantization_loss = F.mse_loss(pred, quantized_pred)

        # Combine losses
        total_loss = reconstruction_loss + 0.5 * edge_loss + 0.3 * quantization_loss

        return total_loss


class EdgeDetectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3),
        )

        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3),
        )

    def forward(self, x):
        # Convert to grayscale
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)

        # Apply Sobel filters
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        # Calculate gradient magnitude
        edge_map = torch.sqrt(grad_x**2 + grad_y**2)
        return edge_map


class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        # Create log directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize CSV file with headers
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"training_stats_{timestamp}.csv"
        self.stats = []

        # Write headers
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "learning_rate", "time_elapsed"]
            )

        self.start_time = time.time()

    def log_epoch(self, epoch, train_loss, val_loss, learning_rate):
        time_elapsed = time.time() - self.start_time
        stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "time_elapsed": time_elapsed,
        }
        self.stats.append(stats)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, learning_rate, time_elapsed])

    def plot_training_curves(self):
        df = pd.DataFrame(self.stats)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot losses
        ax1.plot(df["epoch"], df["train_loss"], label="Training Loss", marker="o")
        ax1.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="o")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss Over Time")
        ax1.legend()
        ax1.grid(True)

        # Plot learning rate
        ax2.plot(
            df["epoch"],
            df["learning_rate"],
            label="Learning Rate",
            marker="o",
            color="green",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Over Time")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = self.log_dir / "training_curves.png"
        plt.savefig(plot_path)
        plt.close()

        return plot_path


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    patience=5,
):
    logger = TrainingLogger()
    best_val_loss = float("inf")
    patience_counter = 0

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (ram_view, target_image) in enumerate(progress_bar):
            ram_view, target_image = ram_view.to(device), target_image.to(device)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = model(ram_view)
                loss = criterion(output, target_image)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Validation with mixed precision
        model.eval()
        val_loss = 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for ram_view, target_image in val_loader:
                ram_view, target_image = ram_view.to(device), target_image.to(device)
                output = model(ram_view)
                loss = criterion(output, target_image)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                "best_model.pth",
            )
            print("Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logger.log_epoch(
            epoch + 1, avg_train_loss, avg_val_loss, optimizer.param_groups[0]["lr"]
        )
        logger.plot_training_curves()

    return model


def setup_training(db_path):
    dataset = GameBoyDataset(db_path)

    # Stratified split would be better if possible
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

    model = ImprovedGBCModel()

    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Combined loss function
    criterion = nn.L1Loss().to(device)  # L1 loss often works better for images

    # Improved optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    # More sophisticated learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    return (
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
    )


# Main execution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WRAM to Image Model")
    parser.add_argument(
        "--db_path", type=str, required=True, help="Path to the SQLite database file"
    )
    args = parser.parse_args()

    db_path = args.db_path

    (
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
    ) = setup_training(db_path)
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=50,
        device=device,
    )
