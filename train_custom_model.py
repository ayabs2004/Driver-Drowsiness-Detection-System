"""
train_custom_model.py – CNN Model Training
==========================================
Fixes applied:
  - val_dir now correctly points to 'valid' (not 'test')
  - stronger augmentation to prevent memorization
  - dropout reduced to 0.4 to work properly with BatchNorm
  - class weights added back carefully to handle any remaining imbalance
  - best checkpoint (not final epoch) is loaded and saved as the final model
"""

import os
import numpy as np
import keras
import tensorflow as tf
from sklearn.utils import class_weight as skl_cw
from model_architecture import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE       = (96, 96)
BATCH_SIZE     = 32
EPOCHS         = 50
BASE_DIR       = r"c:\Users\aya\Documents\driver ai project\processed_data"
MODEL_SAVE     = r"c:\Users\aya\Documents\driver ai project\drowsiness_model.keras"
BEST_CKPT      = r"c:\Users\aya\Documents\driver ai project\best_checkpoint.keras"
TARGET_CLASSES = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']


def train():
    # FIX 1: use 'valid' not 'test' for validation
    train_dir = os.path.join(BASE_DIR, "train")
    val_dir   = os.path.join(BASE_DIR, "valid")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}\n"
            "Run data_preparation.py first."
        )

    # ── Augmentation ──────────────────────────────────────────────────────
    # FIX 2: stronger augmentation forces the model to generalize,
    # not memorize pixel patterns from tight 96x96 crops
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.6, 1.4],   # simulate different lighting
        channel_shift_range=20.0,       # simulate color variation
        horizontal_flip=True,
        fill_mode='nearest',
    )

    # Validation: ONLY rescale — no augmentation on val
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=TARGET_CLASSES,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=TARGET_CLASSES,
        shuffle=False,
    )

    print(f"\nTraining samples   : {train_gen.samples}")
    print(f"Validation samples : {val_gen.samples}")
    print(f"Class indices      : {train_gen.class_indices}\n")

    # ── Class weights ──────────────────────────────────────────────────────
    # Compute from actual file counts so any residual imbalance is handled
    labels = train_gen.classes
    weights = skl_cw.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels,
    )
    class_weights = dict(enumerate(weights))
    print(f"Class weights: {class_weights}\n")

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',     # watch accuracy, not just loss
            patience=10,                # give it more time to improve
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_CKPT,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Build & train ──────────────────────────────────────────────────────
    # FIX 3: dropout lowered to 0.4 inside build_model — edit model_architecture.py
    # or override here by rebuilding with a patched version
    model = build_model(input_shape=(96, 96, 3), num_classes=4)
    model.summary()

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # FIX: save the best checkpoint as the final model (not the last epoch)
    best_model = keras.models.load_model(BEST_CKPT)
    best_model.save(MODEL_SAVE)
    print(f"\n[OK] Best model saved to: {MODEL_SAVE}")

    # ── Summary ────────────────────────────────────────────────────────────
    best_val_acc  = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    final_acc     = max(history.history['accuracy'])
    gap           = final_acc - best_val_acc

    print(f"\n{'='*50}")
    print(f"Best val_accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
    print(f"Best val_loss     : {best_val_loss:.4f}")
    print(f"Train/val gap     : {gap:.4f}  {'⚠ Overfitting' if gap > 0.15 else '✔ OK'}")
    print(f"{'='*50}")

    if best_val_acc < 0.60:
        print("\n⚠ val_accuracy still below 60%.")
        print("  → Check that data_preparation.py ran cleanly and all 4 class folders have samples.")
        print("  → Run: python data_preparation.py  then  python train_custom_model.py")
    elif best_val_acc < 0.80:
        print("\n⚠ val_accuracy is acceptable but not great.")
        print("  → Consider collecting more real driving images for the weaker classes.")
    else:
        print("\n✔ Training looks good. Run app.py to test.")


if __name__ == "__main__":
    train()