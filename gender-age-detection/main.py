"""
Gender & Age Detection using Deep Learning
===========================================

Tech Stack : Python, OpenCV DNN, Caffe pre-trained CNNs
Models     : Levi & Hassner (Adience benchmark)

Usage:
  python main.py                   # webcam (live)
  python main.py --image photo.jpg # single image
"""

import argparse
import os
import sys
import cv2
import numpy as np

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FACE_PROTO = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO  = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL  = os.path.join(MODEL_DIR, "age_net.caffemodel")
GEN_PROTO  = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GEN_MODEL  = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]

GENDER_LIST = ["Male", "Female"]

CONFIDENCE_THRESHOLD = 0.7

# Colours (BGR)
BOX_COLOR   = (0, 255, 0)
TEXT_COLOR  = (255, 255, 255)
BG_COLOR    = (0, 0, 0)

PADDING = 20  # pixels to pad the face crop


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def check_models() -> None:
    """Verify that all model files exist."""
    for path in (FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GEN_PROTO, GEN_MODEL):
        if not os.path.isfile(path):
            print(f"[ERROR] Model file not found: {path}")
            print("        Run 'python download_models.py' first.")
            sys.exit(1)


def load_networks():
    """Load the three DNN networks and return them."""
    face_net   = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net    = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GEN_MODEL, GEN_PROTO)
    return face_net, age_net, gender_net


def highlight_face(net, frame, threshold=CONFIDENCE_THRESHOLD):
    """
    Detect faces in *frame* using the SSD face detector.
    Returns a list of bounding-box tuples (x1, y1, x2, y2).
    """
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


def predict_age_gender(frame, face_box, age_net, gender_net):
    """
    Crop the face from *frame* using *face_box*, run it through
    the age and gender networks, and return the predictions.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face_box

    # Add padding around the face
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(w, x2 + PADDING)
    y2 = min(h, y2 + PADDING)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227),
                                 MODEL_MEAN, swapRB=False)

    # Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    gender_conf = gender_preds[0].max()

    # Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    age_conf = age_preds[0].max()

    return (gender, gender_conf), (age, age_conf)


def draw_label(frame, text, x, y):
    """Draw a label with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Background rectangle
    cv2.rectangle(frame,
                  (x, y - th - 10),
                  (x + tw + 10, y + baseline - 5),
                  BG_COLOR, cv2.FILLED)

    # Text
    cv2.putText(frame, text, (x + 5, y - 5),
                font, scale, TEXT_COLOR, thickness)


def annotate_frame(frame, face_net, age_net, gender_net):
    """Detect faces in *frame* and annotate with gender & age."""
    result = frame.copy()
    boxes = highlight_face(face_net, frame)

    if not boxes:
        cv2.putText(result, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

    for box in boxes:
        x1, y1, x2, y2 = box

        gender_info, age_info = predict_age_gender(
            frame, box, age_net, gender_net
        )

        if gender_info is None:
            continue

        gender, g_conf = gender_info
        age, a_conf = age_info

        # Draw bounding box
        color = (255, 0, 128) if gender == "Female" else (255, 200, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Labels
        label = f"{gender} ({g_conf*100:.0f}%), Age: {age} ({a_conf*100:.0f}%)"
        draw_label(result, label, x1, y1)

    return result


# ──────────────────────────────────────────────
# Mode handlers
# ──────────────────────────────────────────────
def run_on_image(image_path, face_net, age_net, gender_net):
    """Run detection on a single image and display the result."""
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

    # Save output
    out_path = os.path.splitext(image_path)[0] + "_output.jpg"
    cv2.imwrite(out_path, result)
    print(f"Output saved to: {out_path}")


def run_on_webcam(face_net, age_net, gender_net):
    """Run real-time detection on the webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    print("=" * 50)
    print("  Gender & Age Detection — Live Webcam")
    print("  Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = annotate_frame(frame, face_net, age_net, gender_net)
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
        description="Gender & Age Detection using Deep Learning (OpenCV DNN)"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to input image. If omitted, webcam is used."
    )
    parser.add_argument(
        "--confidence", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Face detection confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.confidence

    # Check & load models
    check_models()
    face_net, age_net, gender_net = load_networks()
    print("[INFO] Models loaded successfully.\n")

    if args.image:
        run_on_image(args.image, face_net, age_net, gender_net)
    else:
        run_on_webcam(face_net, age_net, gender_net)


if __name__ == "__main__":
    main()
