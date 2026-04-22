# Gender & Age Detection using Deep Learning

> **Mini Project** — LP5 (AI/ML & Deep Learning)

## Tech Stack

| Layer        | Technology                            |
|-------------|--------------------------------------|
| Language     | Python 3.8+                           |
| DL Framework | OpenCV DNN (Caffe backend)            |
| Face Detect  | SSD MobileNet (TensorFlow)            |
| Age Model    | Levi & Hassner CNN (Adience dataset)  |
| Gender Model | Levi & Hassner CNN (Adience dataset)  |

## How It Works

1. **Face Detection** — An SSD-based face detector locates all faces in the frame.
2. **Pre-processing** — Each detected face is cropped, padded, and converted to a 227×227 blob.
3. **Gender Classification** — A CNN predicts `Male` or `Female`.
4. **Age Estimation** — A CNN predicts one of 8 age buckets:
   `(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)`

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download pre-trained models (~45 MB)
python download_models.py
```

## Usage

```bash
# Real-time webcam detection
python main.py

# Detect from an image
python main.py --image path/to/photo.jpg

# Adjust face detection confidence (default 0.7)
python main.py --confidence 0.5
```

## Project Structure

```
gender-age-detection/
├── main.py               # Main detection script
├── download_models.py    # Downloads pre-trained model files
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── models/               # Pre-trained model weights (auto-downloaded)
    ├── opencv_face_detector.pbtxt
    ├── opencv_face_detector_uint8.pb
    ├── age_deploy.prototxt
    ├── age_net.caffemodel
    ├── gender_deploy.prototxt
    └── gender_net.caffemodel
```

## Output

- **Webcam mode**: Live video with bounding boxes, gender labels, and age ranges.
- **Image mode**: Annotated image displayed on screen and saved as `<filename>_output.jpg`.

## Key Concepts (for Viva)

- **CNN (Convolutional Neural Network)**: Extracts spatial features from face images.
- **Transfer Learning**: Uses pre-trained weights from the Adience benchmark.
- **SSD (Single Shot Detector)**: Efficient real-time face detection.
- **Blob**: Pre-processed image tensor fed into the DNN.
- **Softmax Output**: Probability distribution over age/gender classes.

## Controls

| Key | Action       |
|-----|-------------|
| `q` | Quit webcam |

## References

- Levi, G., & Hassner, T. (2015). *Age and Gender Classification Using Convolutional Neural Networks*. IEEE Workshop on AMFG.
- OpenCV DNN Module: https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html
