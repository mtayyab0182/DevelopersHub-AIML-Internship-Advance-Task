"""
model.py
--------
Multimodal architecture:
    Image  -> CNN (ResNet18 backbone) -> image embedding
    Tabular -> MLP                    -> tabular embedding
    [image embedding ; tabular embedding] -> fusion MLP -> price
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """CNN feature extractor built on a pretrained ResNet18."""

    def __init__(self, output_dim=128, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()  # strip the classification head, keep 512-d features
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projection(features)


class TabularEncoder(nn.Module):
    """Small MLP that turns structured features into an embedding."""

    def __init__(self, input_dim, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalHousingModel(nn.Module):
    """Fuses image + tabular embeddings and regresses to a price."""

    def __init__(self, tabular_input_dim, image_feat_dim=128, tabular_feat_dim=32):
        super().__init__()
        self.image_encoder = ImageEncoder(output_dim=image_feat_dim)
        self.tabular_encoder = TabularEncoder(tabular_input_dim, output_dim=tabular_feat_dim)

        fusion_dim = image_feat_dim + tabular_feat_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_encoder(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(fused).squeeze(1)
