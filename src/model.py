"""
Crack detection model — EfficientNetB0 + custom head.

Input pipeline produces images normalised to [0, 1].
EfficientNetB0 internally expects [0, 255], so a Rescaling(255) layer
sits at the model input to bridge the two.

Two-phase training strategy
─────────────────────────────
Phase 1 — Feature extraction : backbone frozen, head trains at LR_PHASE1.
Phase 2 — Fine-tuning        : top FINE_TUNE_LAYERS of backbone unfrozen,
                               full model trains at LR_PHASE2.
"""

import tensorflow as tf
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
PROJECT_DIR      = Path(r"C:\Projects\infrastructure-defect-detection")
OUTPUTS_DIR      = PROJECT_DIR / "outputs"

IMG_SHAPE        = (227, 227, 3)
DROPOUT_RATE     = 0.3
LR_PHASE1        = 1e-3   # head-only training
LR_PHASE2        = 1e-5   # fine-tuning
FINE_TUNE_LAYERS = 20     # top N backbone layers to unfreeze in phase 2

# ── Model builder ────────────────────────────────────────────────────────────
def build_model(trainable_base: bool = False) -> tf.keras.Model:
    """
    Build an EfficientNetB0 transfer-learning model for binary crack detection.

    Args:
        trainable_base: False → backbone frozen (phase 1).
                        True  → backbone fully unfrozen (phase 2 start point,
                                use unfreeze_top_layers() for partial unfreezing).
    Returns:
        Uncompiled tf.keras.Model.
    """
    inputs = tf.keras.Input(shape=IMG_SHAPE, name="input_image")

    # Our pipeline outputs [0, 1]; EfficientNetB0 expects [0, 255].
    x = tf.keras.layers.Rescaling(255.0, name="rescale_to_255")(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE,
    )
    backbone.trainable = trainable_base
    # training=False keeps backbone BatchNorm layers in inference mode while frozen.
    x = backbone(x, training=False)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return tf.keras.Model(inputs, outputs, name="crack_detector")


def compile_model(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int = FINE_TUNE_LAYERS) -> tf.keras.Model:
    """
    Unfreeze the top n_layers of the EfficientNetB0 backbone for phase-2 fine-tuning.
    Call compile_model() again with LR_PHASE2 after this.
    """
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    trainable   = sum(1 for l in backbone.layers if l.trainable)
    frozen      = sum(1 for l in backbone.layers if not l.trainable)
    print(f"  Backbone — total: {len(backbone.layers)}, trainable: {trainable}, frozen: {frozen}")
    return model


# ── Main: build, summarise, save ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🏗️  Building model (phase 1 — backbone frozen)...")
    model = build_model(trainable_base=False)
    model = compile_model(model, lr=LR_PHASE1)
    model.summary()

    # Save text summary
    summary_path = OUTPUTS_DIR / "model_summary.txt"
    with open(summary_path, "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    print(f"\n💾 Model summary saved to {summary_path}")

    # Parameter breakdown
    trainable_params     = sum(w.numpy().size for w in model.trainable_weights)
    non_trainable_params = sum(w.numpy().size for w in model.non_trainable_weights)
    print(f"\n  Trainable params     : {trainable_params:,}")
    print(f"  Non-trainable params : {non_trainable_params:,}")
    print(f"  Total params         : {trainable_params + non_trainable_params:,}")

    print("\n✅ model.py ready.")
