"""
Download pre-trained models for Gender & Age Detection.

Models used:
  - Face Detection : OpenCV SSD (Caffe)
  - Age Estimation : Levi & Hassner CNN (Caffe)
  - Gender Classification : Levi & Hassner CNN (Caffe)

Run this script once before running main.py.
"""

import os
import urllib.request
import sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

FILES = {
    # Face detection model
    "opencv_face_detector.pbtxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    ),
    "opencv_face_detector_uint8.pb": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb"
    ),
    # Age prediction model
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/refs/heads/master/models/age_net.caffemodel"
    ),
    # Gender prediction model
    "gender_deploy.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/refs/heads/master/models/gender_net.caffemodel"
    ),
}


def download_file(url: str, dest: str) -> None:
    """Download a file from *url* to *dest* with a progress indicator and retries."""
    import time
    if os.path.exists(dest):
        print(f"  [SKIP] {os.path.basename(dest)} already exists.")
        return

    print(f"  [DOWN] {os.path.basename(dest)} ...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(dest, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"         Done ({size_mb:.2f} MB)")
            return
        except Exception as e:
            print(f"  [WARN] Attempt {attempt+1} failed for {os.path.basename(dest)}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f"  [FAIL] Could not download {os.path.basename(dest)} after multiple attempts.")
    if os.path.exists(dest):
        os.remove(dest)
    sys.exit(1)


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading models to: {MODEL_DIR}\n")

    for filename, url in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        download_file(url, dest)

    print("\nAll models downloaded successfully!")
    print("You can now run:  python main.py")


if __name__ == "__main__":
    main()
