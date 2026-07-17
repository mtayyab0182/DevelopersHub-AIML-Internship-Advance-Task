# 🏠 Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

Predicts house prices by fusing **structured tabular features** (sqft, bedrooms, bathrooms, lot size, year built) with **visual features extracted from house photos** using a CNN.

## Architecture

```
House image ──► ResNet18 (pretrained, fc stripped) ──► 128-d image embedding ─┐
                                                                                ├─► concat ──► MLP ──► price
Tabular row ──► MLP (5 features → 64 → 32) ──► 32-d tabular embedding ────────┘
```

## Project structure

```
Task-3-Multimodal-Housing-Price-Prediction/
├── generate_synthetic_data.py   # Creates a test dataset (CSV + images) — no real data needed
├── dataset.py                    # PyTorch Dataset pairing tabular rows with images
├── model.py                      # CNN + tabular fusion architecture
├── train.py                      # Training loop, reports MAE/RMSE each epoch
├── evaluate.py                   # Final evaluation + predicted-vs-actual scatter plot
├── predict.py                    # Run a prediction on a single house
├── requirements.txt
├── .gitignore
└── README.md
```

## Skills demonstrated

- Multimodal machine learning (image + tabular fusion)
- Convolutional Neural Networks (transfer learning with ResNet18)
- Feature fusion between modalities
- Regression modeling and evaluation (MAE, RMSE)

## 1. Setup

```bash
cd Task-3-Multimodal-Housing-Price-Prediction
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

## 2. Get a dataset

You have two options:

### Option A — Quick test with synthetic data (recommended first run)

No download needed. This generates a CSV of housing features plus matching synthetic house images so you can confirm the whole pipeline runs end-to-end before using real data:

```bash
python generate_synthetic_data.py --n_samples 500 --output_dir ./data
```

This creates:
```
data/
├── houses.csv
└── images/
    ├── house_0000.jpg
    ├── house_0001.jpg
    └── ...
```

### Option B — Real dataset

A well-known public dataset for this exact task is the **"Houses and Images" dataset (Ahmed & Moustafa)**, available on Kaggle — search *"House Prices and Images SoCal"* or *"Housing Price + Images dataset"*. It requires a free Kaggle account and manual download/`kaggle datasets download` since Kaggle requires authentication.

To use your own or a downloaded dataset, format it to match the expected CSV schema:

| column | type | description |
|---|---|---|
| sqft | float | square footage |
| bedrooms | int | number of bedrooms |
| bathrooms | int | number of bathrooms |
| lot_size | float | lot size |
| year_built | int | year built |
| image_path | string | path to the house image, relative to `--image_dir` |
| price | float | target variable |

Place it as `./data/houses.csv` with images under `./data/images/`, matching the synthetic layout above.

## 3. Train

```bash
python train.py --csv_path ./data/houses.csv --image_dir ./data --epochs 15 --batch_size 16
```

Each epoch prints training loss plus validation **MAE** and **RMSE**. The best checkpoint (lowest val MAE) saves to `./checkpoints/best_model.pt`, and the feature scaler saves to `./checkpoints/scaler.joblib` (needed later for evaluation/inference — it keeps tabular features on the same scale the model was trained on).

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | 15 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--val_split` | 0.2 | Fraction held out for validation |

## 4. Evaluate

```bash
python evaluate.py --csv_path ./data/houses.csv --image_dir ./data \
    --checkpoint ./checkpoints/best_model.pt --scaler ./checkpoints/scaler.joblib
```

Prints final MAE/RMSE and saves a predicted-vs-actual scatter plot to `./checkpoints/predicted_vs_actual.png`.

## 5. Predict a single house

```bash
python predict.py --image ./data/images/house_0001.jpg \
    --sqft 2200 --bedrooms 3 --bathrooms 2 --lot_size 5000 --year_built 2005
```

## Results

| Metric | Score |
|---|---|
| MAE | *fill in after training* |
| RMSE | *fill in after training* |

## Notes

- `ResNet18` weights download automatically from `torchvision` on first run (cached locally afterward).
- Swap in `resnet34`/`resnet50` in `model.py` for a stronger (slower) image encoder.
- The scaler is fit on the full dataset for simplicity here; for a stricter evaluation, fit it only on the training split before scoring validation/test data.

## License

MIT
