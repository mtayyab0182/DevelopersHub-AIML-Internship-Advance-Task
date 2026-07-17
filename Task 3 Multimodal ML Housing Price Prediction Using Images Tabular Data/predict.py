"""
predict.py
----------
Run a prediction for a single house given its tabular features and an image.

Usage:
    python predict.py --image ./data/images/house_0001.jpg \
        --sqft 2200 --bedrooms 3 --bathrooms 2 --lot_size 5000 --year_built 2005
"""

import argparse
import torch
import joblib
import numpy as np
from PIL import Image

from dataset import IMAGE_TRANSFORM, NUMERIC_COLUMNS
from model import MultimodalHousingModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--sqft", type=float, required=True)
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=int, required=True)
    parser.add_argument("--lot_size", type=float, required=True)
    parser.add_argument("--year_built", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--scaler", type=str, default="./checkpoints/scaler.joblib")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = joblib.load(args.scaler)
    model = MultimodalHousingModel(tabular_input_dim=len(NUMERIC_COLUMNS)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    raw_features = np.array(
        [[args.sqft, args.bedrooms, args.bathrooms, args.lot_size, args.year_built]]
    )
    scaled = scaler.transform(raw_features).astype(np.float32)
    tabular_tensor = torch.tensor(scaled).to(device)

    with torch.no_grad():
        prediction = model(image_tensor, tabular_tensor).item()

    print(f"Predicted price: ${prediction:,.2f}")


if __name__ == "__main__":
    main()
