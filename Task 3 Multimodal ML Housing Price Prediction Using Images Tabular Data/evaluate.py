"""
evaluate.py
-----------
Loads a trained checkpoint and reports MAE / RMSE on a held-out CSV,
plus saves a predicted-vs-actual scatter plot.

Usage:
    python evaluate.py --csv_path ./data/houses.csv --image_dir ./data \
        --checkpoint ./checkpoints/best_model.pt --scaler ./checkpoints/scaler.joblib
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

from dataset import HousingDataset, NUMERIC_COLUMNS
from model import MultimodalHousingModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/houses.csv")
    parser.add_argument("--image_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--scaler", type=str, default="./checkpoints/scaler.joblib")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--plot_path", type=str, default="./checkpoints/predicted_vs_actual.png")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = joblib.load(args.scaler)
    dataset = HousingDataset(args.csv_path, args.image_dir, scaler=scaler, fit_scaler=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MultimodalHousingModel(tabular_input_dim=len(NUMERIC_COLUMNS)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
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

    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5)
    lims = [min(targets + preds), max(targets + preds)]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title(f"Predicted vs Actual (MAE={mae:,.0f}, RMSE={rmse:,.0f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_path)
    print(f"Saved scatter plot to {args.plot_path}")


if __name__ == "__main__":
    main()
