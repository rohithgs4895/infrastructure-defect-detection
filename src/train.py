"""
Training script for crack detection model.

Pre-requisites:
  Run preprocessing.py first to generate data/processed/{train,val,test}.csv

Two-phase strategy (defined in model.py):
  Phase 1 — backbone frozen, head trains at LR_PHASE1 for up to EPOCHS_P1 epochs.
  Phase 2 — top FINE_TUNE_LAYERS backbone layers unfrozen, trains at LR_PHASE2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))   # ensure model.py is importable

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from model import (
    build_model, compile_model, unfreeze_top_layers,
    LR_PHASE1, LR_PHASE2, FINE_TUNE_LAYERS,
)

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Projects\infrastructure-defect-detection")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR    = PROJECT_DIR / "models"
OUTPUTS_DIR   = PROJECT_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = (227, 227)
BATCH_SIZE = 32
SEED       = 42
EPOCHS_P1  = 10
EPOCHS_P2  = 10

# ── 1. Load split CSVs ───────────────────────────────────────────────────────
print("📂 Loading split CSVs...")
train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")
test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")
print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

# ── 2. tf.data pipelines ─────────────────────────────────────────────────────
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

def load_image(path, label):
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_and_augment(path, label):
    image, label = load_image(path, label)
    image = tf.clip_by_value(augmenter(image, training=True), 0.0, 1.0)
    return image, label

def make_dataset(df, augmented, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("int32"))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    fn = load_and_augment if augmented else load_image
    return ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("⚙️  Building tf.data pipelines...")
train_ds = make_dataset(train_df, augmented=True,  shuffle=True)
val_ds   = make_dataset(val_df,   augmented=False, shuffle=False)
test_ds  = make_dataset(test_df,  augmented=False, shuffle=False)

# ── 3. Callbacks ─────────────────────────────────────────────────────────────
def make_callbacks(phase: int):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"best_phase{phase}.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUTS_DIR / f"history_phase{phase}.csv"),
            append=False,
        ),
    ]

# ── 4. Phase 1: feature extraction (backbone frozen) ─────────────────────────
print(f"\n🚀 Phase 1 — Feature extraction  (backbone frozen, lr={LR_PHASE1})")
model = build_model(trainable_base=False)
model = compile_model(model, lr=LR_PHASE1)

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P1,
    callbacks=make_callbacks(phase=1),
)

# ── 5. Phase 2: fine-tuning (top N backbone layers unfrozen) ─────────────────
print(f"\n🔓 Phase 2 — Fine-tuning  (top {FINE_TUNE_LAYERS} backbone layers, lr={LR_PHASE2})")
model = unfreeze_top_layers(model, n_layers=FINE_TUNE_LAYERS)
model = compile_model(model, lr=LR_PHASE2)

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P2,
    callbacks=make_callbacks(phase=2),
)

# ── 6. Save final model ───────────────────────────────────────────────────────
final_path = MODELS_DIR / "crack_detector_final.keras"
model.save(str(final_path))
print(f"\n💾 Final model saved → {final_path}")

# ── 7. Evaluate on test set ───────────────────────────────────────────────────
print("\n📊 Test-set evaluation...")
results = model.evaluate(test_ds, verbose=1)
print("\n  " + "  ".join(f"{k}: {v:.4f}" for k, v in zip(model.metrics_names, results)))

# ── 8. Plot training curves ───────────────────────────────────────────────────
def plot_metric(h1, h2, metric, out_path):
    v1 = h1.history[metric];      val_v1 = h1.history[f"val_{metric}"]
    v2 = h2.history[metric];      val_v2 = h2.history[f"val_{metric}"]
    ep1 = range(1, len(v1) + 1)
    ep2 = range(len(v1) + 1, len(v1) + len(v2) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep1, v1,     "b-",  label="Train P1")
    ax.plot(ep1, val_v1, "b--", label="Val   P1")
    ax.plot(ep2, v2,     "r-",  label="Train P2")
    ax.plot(ep2, val_v2, "r--", label="Val   P2")
    ax.axvline(len(v1) + 0.5, color="gray", linestyle=":", label="P1 → P2")
    ax.set_title(metric.capitalize())
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

print("\n📈 Saving training plots...")
for metric in ("loss", "accuracy", "auc"):
    plot_metric(history_p1, history_p2, metric, OUTPUTS_DIR / f"history_{metric}.png")

print("\n✅ Training complete.")
