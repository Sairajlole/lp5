"""
Gender & Age Detection using Deep Learning  (v3 — Auto-trained model)
======================================================================

Uses the custom-trained Keras model (from train.py) if available,
otherwise falls back to the pre-trained Caffe models.

Usage:
  python main.py                   # webcam (live)
  python main.py --image photo.jpg # single image
"""

import argparse
import os
import sys
import cv2
import numpy as np
from collections import deque

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
TRAINED_DIR = os.path.join(BASE_DIR, "trained_model")

FACE_PROTO  = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL  = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO   = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL   = os.path.join(MODEL_DIR, "age_net.caffemodel")
GEN_PROTO   = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GEN_MODEL   = os.path.join(MODEL_DIR, "gender_net.caffemodel")
KERAS_MODEL = os.path.join(TRAINED_DIR, "age_gender_model.keras")
KERAS_MODEL_FINAL = os.path.join(TRAINED_DIR, "age_gender_model_final.keras")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]
AGE_MIDPOINTS = np.array([1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0])

GENDER_LIST = ["Male", "Female"]

CONFIDENCE_THRESHOLD = 0.7
PADDING = 20
SMOOTHING_WINDOW = 5

# Will be set at runtime depending on which model is loaded
KERAS_IMG_SIZE = 128  # must match train.py IMG_SIZE
USE_KERAS_MODEL = False
keras_model = None


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
def check_face_model():
    """Verify face detection model files exist."""
    for path in (FACE_PROTO, FACE_MODEL):
        if not os.path.isfile(path):
            print(f"[ERROR] Face model file not found: {path}")
            print("        Run 'python download_models.py' first.")
            sys.exit(1)


def load_networks():
    """
    Load face detector + age/gender model.
    Tries the custom-trained Keras model first, falls back to Caffe.
    """
    global USE_KERAS_MODEL, keras_model

    check_face_model()
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

    # ── Try to load custom-trained Keras model ──
    keras_path = None
    if os.path.isfile(KERAS_MODEL):
        keras_path = KERAS_MODEL
    elif os.path.isfile(KERAS_MODEL_FINAL):
        keras_path = KERAS_MODEL_FINAL

    if keras_path:
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            print(f"[INFO] Loading custom-trained model: {keras_path}")
            keras_model = tf.keras.models.load_model(keras_path)
            USE_KERAS_MODEL = True
            print("[INFO] ✓ Custom-trained model loaded — using PRECISE predictions")
            return face_net, None, None
        except Exception as e:
            print(f"[WARN] Could not load Keras model ({e}), falling back to Caffe models")

    # ── Fallback to Caffe models ──
    print("[INFO] Custom-trained model not found — using pre-trained Caffe models")
    print("[INFO] For better accuracy, run:  python train.py")
    for path in (AGE_PROTO, AGE_MODEL, GEN_PROTO, GEN_MODEL):
        if not os.path.isfile(path):
            print(f"[ERROR] Model file not found: {path}")
            print("        Run 'python download_models.py' first.")
            sys.exit(1)

    age_net    = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GEN_MODEL, GEN_PROTO)
    USE_KERAS_MODEL = False
    return face_net, age_net, gender_net


# ──────────────────────────────────────────────
# Face Detection
# ──────────────────────────────────────────────
def highlight_face(net, frame, threshold=CONFIDENCE_THRESHOLD):
    """Detect faces using the SSD face detector."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2, y2))
    return boxes


# ──────────────────────────────────────────────
# Prediction — Keras (custom-trained)
# ──────────────────────────────────────────────
def predict_keras(frame, face_box):
    """Predict using the custom-trained Keras model (precise single-year age)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face_box

    x1p = max(0, x1 - PADDING)
    y1p = max(0, y1 - PADDING)
    x2p = min(w, x2 + PADDING)
    y2p = min(h, y2 + PADDING)

    face_crop = frame[y1p:y2p, x1p:x2p]
    if face_crop.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (KERAS_IMG_SIZE, KERAS_IMG_SIZE))
    face_norm = face_resized.astype(np.float32) / 255.0
    face_batch = np.expand_dims(face_norm, axis=0)

    gender_pred, age_pred = keras_model.predict(face_batch, verbose=0)

    gender_prob = float(gender_pred[0][0])
    gender = "Female" if gender_prob > 0.5 else "Male"
    gender_conf = gender_prob if gender == "Female" else (1 - gender_prob)

    age = float(age_pred[0][0]) * 120  # denormalize
    age = max(0, min(120, age))

    return gender, gender_conf, age


# ──────────────────────────────────────────────
# Prediction — Caffe (pre-trained, fallback)
# ──────────────────────────────────────────────
def predict_caffe(frame, face_box, age_net, gender_net):
    """Predict using pre-trained Caffe models with weighted-midpoint age."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face_box

    x1p = max(0, x1 - PADDING)
    y1p = max(0, y1 - PADDING)
    x2p = min(w, x2 + PADDING)
    y2p = min(h, y2 + PADDING)

    face_crop = frame[y1p:y2p, x1p:x2p]
    if face_crop.size == 0:
        return None

    blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                 MODEL_MEAN, swapRB=False)

    # Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    gender_conf = float(gender_preds[0].max())

    # Age — weighted midpoint for more precise estimate
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = float(np.dot(age_preds[0], AGE_MIDPOINTS))

    return gender, gender_conf, age


# ──────────────────────────────────────────────
# Temporal Smoothing
# ──────────────────────────────────────────────
class FaceSmoother:
    """Rolling-window smoother for stable webcam output."""
    def __init__(self, window=SMOOTHING_WINDOW):
        self.window = window
        self.histories = {}

    def _quantize(self, x1, y1, x2, y2, step=60):
        cx = ((x1 + x2) // 2) // step * step
        cy = ((y1 + y2) // 2) // step * step
        return (cx, cy)

    def update(self, box, gender, gender_conf, age):
        key = self._quantize(*box)
        if key not in self.histories:
            self.histories[key] = deque(maxlen=self.window)
        self.histories[key].append((gender, gender_conf, age))

        entries = self.histories[key]

        # Smoothed age
        smooth_age = np.mean([e[2] for e in entries])

        # Smoothed gender — majority vote
        male_w = sum(e[1] for e in entries if e[0] == "Male")
        female_w = sum(e[1] for e in entries if e[0] == "Female")
        smooth_gender = "Male" if male_w >= female_w else "Female"
        smooth_conf = max(male_w, female_w) / max(male_w + female_w, 1e-9)

        return smooth_gender, smooth_conf, smooth_age

    def cleanup(self, active_keys):
        stale = [k for k in self.histories if k not in active_keys]
        for k in stale:
            del self.histories[k]


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────
def draw_label(frame, text, x, y):
    """Draw a label with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(frame,
                  (x, y - th - 10),
                  (x + tw + 10, y + baseline - 5),
                  (0, 0, 0), cv2.FILLED)

    cv2.putText(frame, text, (x + 5, y - 5),
                font, scale, (255, 255, 255), thickness)


def annotate_frame(frame, face_net, age_net, gender_net, smoother=None):
    """Detect faces and annotate with gender & age."""
    result = frame.copy()
    boxes = highlight_face(face_net, frame)

    if not boxes:
        cv2.putText(result, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

    active_keys = set()

    for box in boxes:
        x1, y1, x2, y2 = box

        # Use whichever model is available
        if USE_KERAS_MODEL:
            pred = predict_keras(frame, box)
        else:
            pred = predict_caffe(frame, box, age_net, gender_net)

        if pred is None:
            continue

        gender, g_conf, age = pred

        # Temporal smoothing
        if smoother is not None:
            gender, g_conf, age = smoother.update(box, gender, g_conf, age)
            active_keys.add(smoother._quantize(*box))

        # Draw
        color = (255, 0, 128) if gender == "Female" else (255, 200, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        label = f"{gender} ({g_conf*100:.0f}%), Age: {int(round(age))}"
        draw_label(result, label, x1, y1)

    if smoother is not None:
        smoother.cleanup(active_keys)

    return result


# ──────────────────────────────────────────────
# Mode handlers
# ──────────────────────────────────────────────
def run_on_image(image_path, face_net, age_net, gender_net):
    """Run detection on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    print(f"Processing image: {image_path}")
    result = annotate_frame(frame, face_net, age_net, gender_net)

    cv2.imshow("Gender & Age Detection", result)
    print("Press any key to exit ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_path = os.path.splitext(image_path)[0] + "_output.jpg"
    cv2.imwrite(out_path, result)
    print(f"Output saved to: {out_path}")


def run_on_webcam(face_net, age_net, gender_net):
    """Run real-time detection on the webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    smoother = FaceSmoother(window=SMOOTHING_WINDOW)

    model_type = "Custom-Trained Keras" if USE_KERAS_MODEL else "Pre-trained Caffe"
    print("\n" + "=" * 55)
    print(f"  Gender & Age Detection — Live Webcam")
    print(f"  Model: {model_type}")
    print("  Press 'q' to quit")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = annotate_frame(frame, face_net, age_net, gender_net, smoother=smoother)
        cv2.imshow("Gender & Age Detection — Press 'q' to quit", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    global CONFIDENCE_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Gender & Age Detection (auto-selects best available model)"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image. If omitted, webcam is used.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Face detection confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.confidence

    face_net, age_net, gender_net = load_networks()
    print("[INFO] Models loaded successfully.\n")

    if args.image:
        run_on_image(args.image, face_net, age_net, gender_net)
    else:
        run_on_webcam(face_net, age_net, gender_net)


if __name__ == "__main__":
    main()
