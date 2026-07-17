"""
train.py
--------
Trains the multimodal (image + tabular) housing price regression model.

Usage:
    python train.py --csv_path ./data/houses.csv --image_dir ./data --epochs 15
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from dataset import HousingDataset, NUMERIC_COLUMNS
from model import MultimodalHousingModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/houses.csv")
    parser.add_argument("--image_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, tabular, prices in loader:
            images, tabular = images.to(device), tabular.to(device)
            outputs = model(images, tabular).cpu().numpy()
            preds.extend(outputs.tolist())
            targets.extend(prices.numpy().tolist())

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    return mae, rmse


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset (scaler fit on the full set here for simplicity;
    #    for a rigorous setup fit only on the train split)
    full_dataset = HousingDataset(args.csv_path, args.image_dir, fit_scaler=True)
    joblib.dump(full_dataset.scaler, f"{args.output_dir}/scaler.joblib")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model
    model = MultimodalHousingModel(tabular_input_dim=len(NUMERIC_COLUMNS)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mae = float("inf")

    # 3. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, tabular, prices in train_loader:
            images, tabular, prices = images.to(device), tabular.to(device), prices.to(device)

            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_mae, val_rmse = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss(MSE): {train_loss:,.2f} "
            f"| val_MAE: {val_mae:,.2f} | val_RMSE: {val_rmse:,.2f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
            print(f"  -> new best model saved (val_MAE={val_mae:,.2f})")

    print(f"Training complete. Best val MAE: {best_mae:,.2f}")
    print(f"Best model saved to {args.output_dir}/best_model.pt")
    print(f"Scaler saved to {args.output_dir}/scaler.joblib")


if __name__ == "__main__":
    main()
