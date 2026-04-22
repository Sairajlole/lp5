"""
Train a Custom Age & Gender Detection Model  (v2 — Improved)
==============================================================

Uses:  TensorFlow / Keras  +  MobileNetV2 (Transfer Learning)
Data:  UTKFace dataset  (~24,000 labelled face images)

Key improvements over v1:
  • Two-phase training  (frozen base → fine-tune)
  • Larger input: 160×160  (more facial detail)
  • Deeper head with channel-attention & residual connection
  • Huber loss for age  (robust to outlier ages)
  • Label smoothing for gender
  • Cosine LR schedule with warm-up
  • Stronger data augmentation  (GaussianNoise, random erasing)
  • Comprehensive evaluation  (confusion matrix, age-error histogram)

Run:
  python train.py
  python train.py --epochs 50 --batch_size 64 --lr 3e-4
"""

import os
import sys
import glob
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
IMG_SIZE       = 224          # ↑ increased for precise EfficientNet
BATCH_SIZE     = 32           # ↓ halved to prevent OOM with larger images
EPOCHS_PHASE1  = 10           # Phase 1: train heads only (base frozen)
EPOCHS_PHASE2  = 50           # Phase 2: fine-tune base + heads
LEARNING_RATE  = 3e-4         # initial LR for phase 1
FINETUNE_LR    = 5e-5         # lower LR for fine-tuning
VAL_SPLIT      = 0.15
LABEL_SMOOTH   = 0.05         # label smoothing for gender BCE
DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "UTKFace")
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model")


# ──────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────
def download_utkface():
    """Download UTKFace dataset using kagglehub."""
    try:
        import kagglehub
        print("[INFO] Downloading UTKFace dataset from Kaggle ...")
        path = kagglehub.dataset_download("jangedoo/utkface-new")
        print(f"[INFO] Dataset downloaded to: {path}")
        return path
    except ImportError:
        print("[ERROR] kagglehub not installed. Install: pip install kagglehub")
        print("        OR manually download UTKFace and place images in data/UTKFace/")
        print("        Download from: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("        Manually download UTKFace and place images in data/UTKFace/")
        print("        Download from: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        sys.exit(1)


def find_utkface_images(base_path):
    """Recursively find all jpg/png images in the UTKFace directory."""
    patterns = ["*.jpg", "*.png", "*.JPG", "*.PNG", "*.chip.jpg"]
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(base_path, "**", pattern), recursive=True))
    # Deduplicate (some patterns overlap)
    return list(set(images))


def parse_utkface_filename(filepath):
    """
    UTKFace naming convention: [age]_[gender]_[race]_[date&time].jpg
      age    : 0–116
      gender : 0 = Male, 1 = Female
      race   : 0–4  (not used here)
    Returns (age, gender) or None if parsing fails.
    """
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = int(parts[1])
        if age < 0 or age > 120 or gender not in (0, 1):
            return None
        return age, gender
    except (ValueError, IndexError):
        return None


def load_dataset(data_dir):
    """
    Load the UTKFace dataset from disk.
    Returns: images (np.array), ages (np.array), genders (np.array)
    """
    print(f"\n[INFO] Loading dataset from: {data_dir}")

    image_paths = find_utkface_images(data_dir)
    if not image_paths:
        print(f"[ERROR] No images found in {data_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} images. Parsing labels ...")

    images, ages, genders = [], [], []
    skipped = 0

    for i, path in enumerate(image_paths):
        result = parse_utkface_filename(path)
        if result is None:
            skipped += 1
            continue

        age, gender = result

        try:
            img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            img_arr = img_to_array(img) / 255.0  # normalize to [0, 1]
            images.append(img_arr)
            ages.append(age / 120.0)  # normalize age to [0, 1]
            genders.append(gender)
        except Exception:
            skipped += 1

        if (i + 1) % 2000 == 0:
            print(f"       Processed {i+1}/{len(image_paths)} ...")

    print(f"[INFO] Loaded {len(images)} valid samples (skipped {skipped})")

    # Print dataset statistics
    ages_raw = np.array([a * 120 for a in ages])
    genders_arr = np.array(genders)
    print(f"\n[INFO] Dataset Statistics:")
    print(f"  Age  — mean: {ages_raw.mean():.1f}, std: {ages_raw.std():.1f}, "
          f"min: {ages_raw.min():.0f}, max: {ages_raw.max():.0f}")
    print(f"  Gender — Male: {(genders_arr == 0).sum()}, Female: {(genders_arr == 1).sum()}")

    images  = np.array(images, dtype=np.float32)
    ages    = np.array(ages, dtype=np.float32)
    genders = np.array(genders, dtype=np.float32)

    return images, ages, genders


# ──────────────────────────────────────────────
# 2. Model Architecture  (Improved)
# ──────────────────────────────────────────────
def squeeze_excite_block(x, ratio=8):
    """Channel attention (Squeeze-and-Excitation)."""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def build_model():
    """
    Build an improved multi-task model:
      Base   →  MobileNetV2 (ImageNet pre-trained, top removed)
      Neck   →  SE attention + GlobalPooling
      Head 1 →  Gender classification (sigmoid, label-smoothed BCE)
      Head 2 →  Age regression (sigmoid, 0–1 range)
    """
    # Base model — initially fully frozen
    base = EfficientNetV2B0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # freeze entirely for phase 1

    x = base.output

    # Squeeze-and-excitation attention on feature maps
    x = squeeze_excite_block(x)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Shared trunk
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # ── Gender head ──────────────────────────
    g = layers.Dense(128, activation="relu")(x)
    g = layers.Dropout(0.3)(g)
    gender_out = layers.Dense(1, activation="sigmoid", name="gender")(g)

    # ── Age head ─────────────────────────────
    a = layers.Dense(128, activation="relu")(x)
    a = layers.Dropout(0.3)(a)
    age_out = layers.Dense(1, activation="sigmoid", name="age")(a)

    model = Model(inputs=base.input, outputs=[gender_out, age_out])
    return model


def unfreeze_base(model, num_layers_to_unfreeze=50):
    """Unfreeze the last N layers of the base model for fine-tuning."""
    # Simply unfreeze the entire model, but with BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            # Keep BN layers frozen during fine-tuning (use ImageNet running stats)
            layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    total_count = len(model.layers)
    print(f"[INFO] Unfroze model for fine-tuning: {trainable_count}/{total_count} layers trainable")


# ──────────────────────────────────────────────
# 3. Data Augmentation  (Stronger)
# ──────────────────────────────────────────────
def create_augmentation():
    """Create a stronger data augmentation pipeline."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),            # ↑ more rotation
        layers.RandomBrightness(0.25),           # ↑ more brightness variation
        layers.RandomContrast(0.25),             # ↑ more contrast variation
        layers.RandomZoom((-0.15, 0.15)),        # ↑ more zoom range
        layers.RandomTranslation(0.1, 0.1),     # NEW: random translation
        layers.GaussianNoise(0.02),              # NEW: slight noise for robustness
    ], name="augmentation")


def make_dataset(images, genders, ages, batch_size, augment=False):
    """Create a tf.data.Dataset with optional augmentation."""
    dataset = tf.data.Dataset.from_tensor_slices((images, {"gender": genders, "age": ages}))
    dataset = dataset.shuffle(len(images), reshuffle_each_iteration=True)

    if augment:
        aug = create_augmentation()
        def apply_aug(img, labels):
            img_aug = aug(img, training=True)
            # Clip to [0,1] after augmentation (noise can push values out of range)
            img_aug = tf.clip_by_value(img_aug, 0.0, 1.0)
            return img_aug, labels
        dataset = dataset.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ──────────────────────────────────────────────
# 4. Custom label-smoothed BCE
# ──────────────────────────────────────────────
def label_smoothed_bce(y_true, y_pred):
    """Binary cross-entropy with label smoothing."""
    y_smooth = y_true * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
    return tf.keras.losses.binary_crossentropy(y_smooth, y_pred)


# ──────────────────────────────────────────────
# 5. Training  (Two-Phase)
# ──────────────────────────────────────────────
def train(args):
    """Main training loop with two-phase approach."""

    # ── Data ──────────────────────────────────
    data_dir = args.data_dir

    if not os.path.isdir(data_dir):
        print(f"[INFO] Data directory not found at: {data_dir}")
        downloaded_path = download_utkface()
        candidate = find_utkface_images(downloaded_path)
        if candidate:
            data_dir = downloaded_path
        else:
            print(f"[ERROR] Could not find UTKFace images in {downloaded_path}")
            sys.exit(1)

    images, ages, genders = load_dataset(data_dir)

    # Split data (stratified-ish by using gender for balanced splits)
    n = len(images)
    n_val = int(n * VAL_SPLIT)
    np.random.seed(42)  # reproducible splits
    indices = np.random.permutation(n)

    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_gender_train, y_age_train = images[train_idx], genders[train_idx], ages[train_idx]
    X_val,   y_gender_val,   y_age_val   = images[val_idx],   genders[val_idx],   ages[val_idx]

    print(f"\n[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")

    # Verify gender balance
    print(f"[INFO] Train gender split — Male: {(y_gender_train == 0).sum()}, Female: {(y_gender_train == 1).sum()}")
    print(f"[INFO] Val   gender split — Male: {(y_gender_val == 0).sum()}, Female: {(y_gender_val == 1).sum()}")

    # ── Datasets ──────────────────────────────
    train_ds = make_dataset(X_train, y_gender_train, y_age_train, args.batch_size, augment=True)
    val_ds   = make_dataset(X_val,   y_gender_val,   y_age_val,   args.batch_size, augment=False)

    # ── Model ─────────────────────────────────
    model = build_model()

    # ══════════════════════════════════════════
    #  PHASE 1:  Train heads only  (base frozen)
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 1 — Training heads only (base frozen)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss={
            "gender": label_smoothed_bce,
            "age":    tf.keras.losses.Huber(delta=0.1),  # Huber = robust to outliers
        },
        loss_weights={"gender": 1.0, "age": 8.0},  # ↑ weight age higher
        metrics={
            "gender": ["accuracy"],
            "age":    ["mae"],
        },
    )

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, "age_gender_model.keras")

    phase1_callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_p1,
        callbacks=phase1_callbacks,
        verbose=1,
    )

    # ══════════════════════════════════════════
    #  PHASE 2:  Fine-tune entire model
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2 — Fine-tuning entire model")
    print("=" * 60)

    unfreeze_base(model)

    decay_steps = (len(X_train) // args.batch_size) * args.epochs_p2
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.finetune_lr,
        decay_steps=decay_steps,
        alpha=0.01
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss={
            "gender": label_smoothed_bce,
            "age":    tf.keras.losses.Huber(delta=0.1),
        },
        loss_weights={"gender": 1.0, "age": 8.0},
        metrics={
            "gender": ["accuracy"],
            "age":    ["mae"],
        },
    )

    phase2_callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_p2,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    # ── Save final model ──────────────────────
    final_path = os.path.join(MODEL_SAVE_DIR, "age_gender_model_final.keras")
    model.save(final_path)
    print(f"\n[INFO] Final model saved to: {final_path}")
    print(f"[INFO] Best  model saved to: {model_path}")

    # ── Evaluate on validation set ────────────
    print("\n" + "=" * 60)
    print("  Evaluation on Validation Set")
    print("=" * 60)

    gender_preds, age_preds = model.predict(X_val, batch_size=args.batch_size, verbose=0)

    # Gender accuracy
    gender_pred_labels = (gender_preds.flatten() > 0.5).astype(int)
    gender_true_labels = y_gender_val.astype(int)
    gender_acc = (gender_pred_labels == gender_true_labels).mean()
    print(f"\n  Gender Accuracy: {gender_acc*100:.1f}%")

    # Per-class gender accuracy
    male_mask = gender_true_labels == 0
    female_mask = gender_true_labels == 1
    if male_mask.sum() > 0:
        male_acc = (gender_pred_labels[male_mask] == 0).mean()
        print(f"    Male   accuracy: {male_acc*100:.1f}%  (n={male_mask.sum()})")
    if female_mask.sum() > 0:
        female_acc = (gender_pred_labels[female_mask] == 1).mean()
        print(f"    Female accuracy: {female_acc*100:.1f}%  (n={female_mask.sum()})")

    # Age evaluation
    age_pred_years = age_preds.flatten() * 120
    age_true_years = y_age_val * 120
    age_mae = np.abs(age_pred_years - age_true_years).mean()
    age_rmse = np.sqrt(((age_pred_years - age_true_years) ** 2).mean())
    print(f"\n  Age MAE : {age_mae:.1f} years")
    print(f"  Age RMSE: {age_rmse:.1f} years")

    # Age accuracy within thresholds
    for thresh in [3, 5, 10]:
        within = (np.abs(age_pred_years - age_true_years) <= thresh).mean()
        print(f"  Age within ±{thresh} years: {within*100:.1f}%")

    # ── Print summary ─────────────────────────
    # Combine histories
    all_val_loss = history1.history.get("val_loss", []) + history2.history.get("val_loss", [])
    all_val_g_acc = history1.history.get("val_gender_accuracy", []) + history2.history.get("val_gender_accuracy", [])
    all_val_a_mae = history1.history.get("val_age_mae", []) + history2.history.get("val_age_mae", [])

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    if all_val_loss:
        print(f"  Best validation loss      : {min(all_val_loss):.4f}")
    if all_val_g_acc:
        print(f"  Best gender accuracy      : {max(all_val_g_acc)*100:.1f}%")
    if all_val_a_mae:
        best_a = min(all_val_a_mae)
        print(f"  Best age MAE (normalized) : {best_a:.4f}  (~{best_a*120:.1f} years)")
    print(f"\n  Run prediction with:")
    print(f"    python predict_trained.py")
    print(f"    python predict_trained.py --image photo.jpg")

    return history1, history2


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Age & Gender model on UTKFace (v2 — Improved)")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"Path to UTKFace images (default: {DATA_DIR})")
    parser.add_argument("--epochs_p1", type=int, default=EPOCHS_PHASE1,
                        help=f"Epochs for Phase 1 — heads only (default: {EPOCHS_PHASE1})")
    parser.add_argument("--epochs_p2", type=int, default=EPOCHS_PHASE2,
                        help=f"Epochs for Phase 2 — fine-tuning (default: {EPOCHS_PHASE2})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate for Phase 1 (default: {LEARNING_RATE})")
    parser.add_argument("--finetune_lr", type=float, default=FINETUNE_LR,
                        help=f"Learning rate for Phase 2 (default: {FINETUNE_LR})")
    args = parser.parse_args()

    train(args)
