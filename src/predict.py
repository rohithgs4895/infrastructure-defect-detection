"""
Crack detection inference pipeline.

Usage:
    python src/predict.py --input-dir path/to/images [--threshold 0.5]

Outputs:
    outputs/predictions.csv  — columns: filename, prediction, confidence, label
    Console summary           — total images, cracks found, no-crack count
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR     = Path(r"C:\Projects\infrastructure-defect-detection")
MODEL_PATH      = PROJECT_DIR / "models" / "crack_detector_final.keras"
DEFAULT_OUTPUT  = PROJECT_DIR / "outputs" / "predictions.csv"

IMG_SIZE         = (227, 227)
BATCH_SIZE       = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crack detection inference pipeline.")
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder containing images to classify.",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for Crack vs No Crack (default: 0.5).",
    )
    return parser.parse_args()


# ── Image helpers ─────────────────────────────────────────────────────────────
def collect_images(folder: Path) -> list:
    """Return a sorted list of image paths found in folder (non-recursive)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: Path):
    """
    Load one image with OpenCV, resize, and normalise to [0, 1].
    Returns float32 array of shape (227, 227, 3), or None on failure.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, image_paths: list, threshold: float) -> list:
    """
    Run batched inference over image_paths.
    Returns a list of dicts: filename, prediction, confidence, label.
    """
    results = []
    total   = len(image_paths)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_paths = image_paths[batch_start : batch_start + BATCH_SIZE]
        batch_imgs, valid_paths = [], []

        for p in batch_paths:
            img = load_image(p)
            if img is None:
                print(f"\n  [WARNING] Could not read '{p.name}' — skipped.")
                continue
            batch_imgs.append(img)
            valid_paths.append(p)

        if not batch_imgs:
            continue

        scores = model.predict(np.stack(batch_imgs), verbose=0).flatten()

        for path, score in zip(valid_paths, scores):
            label = 1 if score >= threshold else 0
            results.append({
                "filename":   path.name,
                "prediction": "Crack" if label == 1 else "No Crack",
                "confidence": round(float(score), 4),
                "label":      label,
            })

        done = min(batch_start + BATCH_SIZE, total)
        print(f"  Processed {done}/{total} images...", end="\r")

    print()  # newline after progress line
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args       = parse_args()
    input_dir  = Path(args.input_dir)
    output_csv = Path(args.output) if args.output else DEFAULT_OUTPUT

    if not input_dir.is_dir():
        print(f"ERROR: '{input_dir}' is not a valid directory.")
        sys.exit(1)

    # 1. Load model
    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("  Model loaded.\n")

    # 2. Collect images
    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        sys.exit(0)
    print(f"Found {len(image_paths)} image(s) in '{input_dir}'.")

    # 3. Run inference
    print(f"Running inference  (threshold={args.threshold}) ...")
    results = run_inference(model, image_paths, threshold=args.threshold)

    if not results:
        print("No images could be processed.")
        sys.exit(1)

    # 4. Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results, columns=["filename", "prediction", "confidence", "label"])
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved -> {output_csv}\n")

    # 5. Summary
    total    = len(df)
    cracks   = int((df["label"] == 1).sum())
    no_crack = int((df["label"] == 0).sum())

    print("-" * 42)
    print("  SUMMARY")
    print("-" * 42)
    print(f"  Total images   : {total}")
    print(f"  Cracks found   : {cracks:<6}  ({cracks / total * 100:.1f} %)")
    print(f"  No crack       : {no_crack:<6}  ({no_crack / total * 100:.1f} %)")
    print("-" * 42)


if __name__ == "__main__":
    main()
