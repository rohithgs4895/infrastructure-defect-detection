import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR = Path(r"C:\Projects\infrastructure-defect-detection\data\raw")
POSITIVE_DIR = RAW_DIR / "Positive"
NEGATIVE_DIR = RAW_DIR / "Negative"

# ── 1. Count images ────────────────────────────────────────────────────────
pos_images = list(POSITIVE_DIR.glob("*.*"))
neg_images = list(NEGATIVE_DIR.glob("*.*"))

print(f"✅ Positive (Crack) images    : {len(pos_images)}")
print(f"✅ Negative (No Crack) images : {len(neg_images)}")
print(f"✅ Total images               : {len(pos_images) + len(neg_images)}")

# ── 2. Check image sizes ───────────────────────────────────────────────────
print("\n📐 Checking image dimensions across all images...")
unique_shapes = {"Positive": set(), "Negative": set()}

for label, paths in [("Positive", pos_images), ("Negative", neg_images)]:
    for img_path in paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Could not read: {img_path}")
            continue
        unique_shapes[label].add(img.shape)

for label, shapes in unique_shapes.items():
    print(f"  {label} unique shapes: {shapes}")

# ── 3. Visualize sample images ─────────────────────────────────────────────
print("\n🖼️  Generating sample visualization...")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample Images — Top: Crack | Bottom: No Crack", fontsize=14)

for i, img_path in enumerate(pos_images[:5]):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️  Could not read: {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(img)
    axes[0, i].set_title("Crack")
    axes[0, i].axis("off")

for i, img_path in enumerate(neg_images[:5]):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️  Could not read: {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[1, i].imshow(img)
    axes[1, i].set_title("No Crack")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig(r"C:\Projects\infrastructure-defect-detection\outputs\sample_images.png")
print("✅ Sample visualization saved to outputs/sample_images.png")
plt.show()

# ── 4. Class balance check ─────────────────────────────────────────────────
print("\n📊 Class Balance:")
total = len(pos_images) + len(neg_images)
print(f"  Crack    : {len(pos_images)/total*100:.1f}%")
print(f"  No Crack : {len(neg_images)/total*100:.1f}%")

if len(pos_images) == len(neg_images):
    print("  ✅ Dataset is perfectly balanced — no rebalancing needed!")
else:
    print("  ⚠️  Dataset is imbalanced — we will handle this during training.")
