"""
Real-time Gender & Age Prediction using Custom-Trained Model
=============================================================

Uses:
  - OpenCV SSD face detector  (same fast face detection)
  - Custom EfficientNetV2B0 model  (trained on UTKFace via train.py)

Run:
  python predict_trained.py              # webcam
  python predict_trained.py --image x.jpg  # single image
"""

import os
import sys
import argparse
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
TRAINED_DIR = os.path.join(BASE_DIR, "trained_model")

FACE_PROTO = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
KERAS_MODEL = os.path.join(TRAINED_DIR, "age_gender_model.keras")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE = 224  # must match train.py IMG_SIZE
CONFIDENCE_THRESHOLD = 0.7
PADDING = 20

GENDER_LABELS = {0: "Male", 1: "Female"}
GENDER_COLORS = {
    "Male":   (255, 200, 0),    # cyan-ish
    "Female": (255, 0, 128),    # pink-ish
}


# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
def load_models():
    """Load face detector and trained Keras model."""
    # Face detector
    if not os.path.isfile(FACE_PROTO) or not os.path.isfile(FACE_MODEL):
        print("[ERROR] Face detection model not found. Run: python download_models.py")
        sys.exit(1)
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

    # Trained model
    model_path = KERAS_MODEL
    if not os.path.isfile(model_path):
        # Try the final model
        model_path = os.path.join(TRAINED_DIR, "age_gender_model_final.keras")
    if not os.path.isfile(model_path):
        print("[ERROR] Trained model not found. Run: python train.py")
        sys.exit(1)

    print(f"[INFO] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded successfully!")
    return face_net, model


# ──────────────────────────────────────────────
# Face detection
# ──────────────────────────────────────────────
def detect_faces(face_net, frame, threshold=CONFIDENCE_THRESHOLD):
    """Detect faces using OpenCV SSD detector."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2, y2, confidence))
    return boxes


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict_face(model, frame, box):
    """Run the trained model on a single face crop."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2, _ = box

    # Add padding
    x1p = max(0, x1 - PADDING)
    y1p = max(0, y1 - PADDING)
    x2p = min(w, x2 + PADDING)
    y2p = min(h, y2 + PADDING)

    face_crop = frame[y1p:y2p, x1p:x2p]
    if face_crop.size == 0:
        return None

    # Preprocess for model
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_norm = face_resized.astype(np.float32) / 255.0
    face_batch = np.expand_dims(face_norm, axis=0)

    # Predict
    gender_pred, age_pred = model.predict(face_batch, verbose=0)

    gender_prob = float(gender_pred[0][0])
    gender = "Female" if gender_prob > 0.5 else "Male"
    gender_conf = gender_prob if gender == "Female" else (1 - gender_prob)

    age = float(age_pred[0][0]) * 120  # denormalize
    age = max(0, min(120, age))  # clamp

    return gender, gender_conf, age


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────
def draw_label(frame, text, x, y, color):
    """Draw a styled label on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - 12), (x + tw + 12, y + baseline), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text, (x + 6, y - 4), font, scale, color, thickness)


def annotate_frame(frame, face_net, model):
    """Detect faces and annotate with predictions."""
    result = frame.copy()
    boxes = detect_faces(face_net, frame)

    if not boxes:
        cv2.putText(result, "No face detected", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return result

    for box in boxes:
        x1, y1, x2, y2, face_conf = box

        pred = predict_face(model, frame, box)
        if pred is None:
            continue

        gender, gender_conf, age = pred
        color = GENDER_COLORS.get(gender, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        corner_len = 15
        t = 3
        # Top-left
        cv2.line(result, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(result, (x1, y1), (x1, y1 + corner_len), color, t)
        # Top-right
        cv2.line(result, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(result, (x2, y1), (x2, y1 + corner_len), color, t)
        # Bottom-left
        cv2.line(result, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(result, (x1, y2), (x1, y2 - corner_len), color, t)
        # Bottom-right
        cv2.line(result, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(result, (x2, y2), (x2, y2 - corner_len), color, t)

        # Labels
        gender_text = f"{gender} ({gender_conf*100:.0f}%)"
        age_text = f"Age: {age:.0f}"

        draw_label(result, gender_text, x1, y1 - 5, color)
        draw_label(result, age_text, x1, y2 + 25, (255, 255, 255))

    return result


# ──────────────────────────────────────────────
# Run modes
# ──────────────────────────────────────────────
def run_webcam(face_net, model):
    """Real-time webcam detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  Custom Age & Gender Detection — Live Webcam")
    print("  (Trained EfficientNetV2B0 model)")
    print("  Press 'q' to quit")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = annotate_frame(frame, face_net, model)

        # FPS counter
        cv2.imshow("Age & Gender Detection (Trained Model) — Press 'q' to quit", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path, face_net, model):
    """Single image detection."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    print(f"[INFO] Processing: {image_path}")
    result = annotate_frame(frame, face_net, model)

    cv2.imshow("Age & Gender Detection (Trained Model)", result)
    print("Press any key to exit ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_path = os.path.splitext(image_path)[0] + "_predicted.jpg"
    cv2.imwrite(out_path, result)
    print(f"[INFO] Output saved: {out_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Age & Gender Detection (Custom Trained Model)"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file. If omitted, uses webcam.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Face detection confidence threshold")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.confidence

    face_net, model = load_models()

    if args.image:
        run_image(args.image, face_net, model)
    else:
        run_webcam(face_net, model)
