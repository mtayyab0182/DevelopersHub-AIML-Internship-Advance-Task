"""
dataset.py
----------
PyTorch Dataset that pairs each tabular row (structured housing features)
with its corresponding house image.

Expected CSV columns:
    sqft, bedrooms, bathrooms, lot_size, year_built, image_path, price

`image_path` should be relative to `image_dir`.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

NUMERIC_COLUMNS = ["sqft", "bedrooms", "bathrooms", "lot_size", "year_built"]

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class HousingDataset(Dataset):
    def __init__(self, csv_path, image_dir, scaler=None, fit_scaler=False, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform or IMAGE_TRANSFORM

        from sklearn.preprocessing import StandardScaler

        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            scaler.fit(self.df[NUMERIC_COLUMNS])
        self.scaler = scaler
        self.features = scaler.transform(self.df[NUMERIC_COLUMNS]).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor(self.features[idx], dtype=torch.float32)
        price = torch.tensor(float(row["price"]), dtype=torch.float32)

        return image, tabular, price
