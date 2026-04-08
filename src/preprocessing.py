import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Projects\infrastructure-defect-detection")
RAW_DIR       = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUTS_DIR   = PROJECT_DIR / "outputs"

IMG_SIZE   = (227, 227)
BATCH_SIZE = 32
VAL_SPLIT  = 0.15   # fraction of total
TEST_SPLIT = 0.15   # fraction of total
SEED       = 42

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Build path / label DataFrame ─────────────────────────────────────────
print("📂 Collecting image paths...")
rows = []
for label, folder in [(1, "Positive"), (0, "Negative")]:
    for p in (RAW_DIR / folder).glob("*.*"):
        rows.append({"path": str(p), "label": label})

df = pd.DataFrame(rows)
print(f"  Total : {len(df)}  |  Crack: {df['label'].sum()}  |  No-Crack: {(df['label'] == 0).sum()}")

# ── 2. Stratified train / val / test split ──────────────────────────────────
print("\n✂️  Splitting dataset  (70 / 15 / 15)...")

train_val_df, test_df = train_test_split(
    df, test_size=TEST_SPLIT, stratify=df["label"], random_state=SEED
)
# val is 15 % of the full dataset → ~17.6 % of the remaining train_val pool
val_relative = VAL_SPLIT / (1.0 - TEST_SPLIT)
train_df, val_df = train_test_split(
    train_val_df, test_size=val_relative, stratify=train_val_df["label"], random_state=SEED
)

for name, split in [("Train", train_df), ("Val  ", val_df), ("Test ", test_df)]:
    print(f"  {name}: {len(split):>6}  "
          f"(Crack: {split['label'].sum()}, No-Crack: {(split['label'] == 0).sum()})")

# ── 3. Save CSVs ─────────────────────────────────────────────────────────────
train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
test_df.to_csv(PROCESSED_DIR  / "test.csv",  index=False)
print(f"\n💾 Split CSVs saved to {PROCESSED_DIR}")

# ── 4. tf.data pipeline helpers ──────────────────────────────────────────────
def load_image(path: str, label: int):
    """Read → decode JPEG → resize → normalise to [0, 1]."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

def load_and_augment(path: str, label: int):
    image, label = load_image(path, label)
    image = augmenter(image, training=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def make_dataset(split_df: pd.DataFrame, augmented: bool, shuffle: bool) -> tf.data.Dataset:
    paths  = split_df["path"].values
    labels = split_df["label"].values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(split_df), seed=SEED)
    preprocess_fn = load_and_augment if augmented else load_image
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ── 5. Build datasets and verify ─────────────────────────────────────────────
print("\n⚙️  Building tf.data pipelines...")
train_ds = make_dataset(train_df, augmented=True,  shuffle=True)
val_ds   = make_dataset(val_df,   augmented=False, shuffle=False)
test_ds  = make_dataset(test_df,  augmented=False, shuffle=False)

sample_images, sample_labels = next(iter(train_ds))
print(f"  Train batch — images: {sample_images.shape}, labels: {sample_labels.shape}")
print(f"  Pixel range : [{sample_images.numpy().min():.3f}, {sample_images.numpy().max():.3f}]")
print(f"  Train batches : {len(train_ds)}")
print(f"  Val   batches : {len(val_ds)}")
print(f"  Test  batches : {len(test_ds)}")

# ── 6. Save augmented sample grid ────────────────────────────────────────────
print("\n🖼️  Saving augmented sample grid...")
fig, axes = plt.subplots(2, 8, figsize=(20, 6))
fig.suptitle("Augmented Training Samples — Top: Crack | Bottom: No Crack", fontsize=13)

crack_shown = nocrack_shown = 0
for imgs, lbls in train_ds:
    for img, lbl in zip(imgs.numpy(), lbls.numpy()):
        if lbl == 1 and crack_shown < 8:
            axes[0, crack_shown].imshow(img)
            axes[0, crack_shown].axis("off")
            crack_shown += 1
        elif lbl == 0 and nocrack_shown < 8:
            axes[1, nocrack_shown].imshow(img)
            axes[1, nocrack_shown].axis("off")
            nocrack_shown += 1
    if crack_shown == 8 and nocrack_shown == 8:
        break

axes[0, 0].set_title("Crack",    fontsize=10, loc="left")
axes[1, 0].set_title("No Crack", fontsize=10, loc="left")
plt.tight_layout()
out_path = OUTPUTS_DIR / "augmented_samples.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"✅ Saved to {out_path}")

print("\n✅ Preprocessing complete.")
