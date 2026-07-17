"""
generate_synthetic_data.py
---------------------------
Generates a small synthetic housing dataset (tabular CSV + matching images)
so you can run and test the full pipeline before plugging in a real dataset.

Usage:
    python generate_synthetic_data.py --n_samples 500 --output_dir ./data
"""

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_house_image(sqft, bedrooms, size, rng):
    """Creates a simple synthetic 'house' image whose color/shape loosely
    correlates with the tabular features, purely so the CNN has *something*
    non-random to learn from during a pipeline smoke test."""
    img = Image.new("RGB", (size, size), color=(135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)

    # House body size scales with sqft
    body_w = int(size * min(0.9, 0.3 + sqft / 6000))
    body_h = int(size * min(0.6, 0.2 + sqft / 8000))
    x0 = (size - body_w) // 2
    y0 = size - body_h - 10
    x1 = x0 + body_w
    y1 = size - 10

    wall_color = (
        int(100 + rng.integers(0, 100)),
        int(60 + rng.integers(0, 80)),
        int(40 + rng.integers(0, 60)),
    )
    draw.rectangle([x0, y0, x1, y1], fill=wall_color)

    # Windows scale with bedrooms
    for i in range(int(bedrooms)):
        wx = x0 + 10 + i * 15
        if wx + 8 > x1:
            break
        draw.rectangle([wx, y0 + 10, wx + 8, y0 + 18], fill=(255, 255, 200))

    # Roof
    draw.polygon([(x0 - 5, y0), (x1 + 5, y0), ((x0 + x1) // 2, y0 - 20)], fill=(80, 40, 40))

    return img


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    rows = []
    for i in range(args.n_samples):
        sqft = rng.uniform(600, 4500)
        bedrooms = rng.integers(1, 6)
        bathrooms = rng.integers(1, 4)
        lot_size = rng.uniform(1000, 10000)
        year_built = rng.integers(1950, 2024)

        # Synthetic price formula + noise, so there's a real signal to learn
        price = (
            sqft * 180
            + bedrooms * 8000
            + bathrooms * 6000
            + lot_size * 5
            + (year_built - 1950) * 300
            + rng.normal(0, 15000)
        )
        price = max(price, 20000)

        image_name = f"house_{i:04d}.jpg"
        img = make_house_image(sqft, bedrooms, args.img_size, rng)
        img.save(os.path.join(image_dir, image_name), quality=90)

        rows.append(
            {
                "id": i,
                "sqft": round(sqft, 1),
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "lot_size": round(lot_size, 1),
                "year_built": int(year_built),
                "image_path": os.path.join("images", image_name),
                "price": round(price, 2),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "houses.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")
    print(f"Wrote {len(df)} images to {image_dir}")


if __name__ == "__main__":
    main()
