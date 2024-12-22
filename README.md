# Real-time Facial Recognition System

A real-time facial recognition system using EfficientNet-B0 for feature extraction and ChromaDB for efficient similarity search. The system captures video from a webcam, detects faces, and matches them against a pre-existing database of facial embeddings.

## Features

- Real-time face detection using OpenCV's Haar Cascade Classifier
- Feature extraction using EfficientNet-B0
- Vector similarity search using ChromaDB
- Persistent storage of facial embeddings
- Configurable proximity and similarity thresholds



## Installation

1. Clone the repository:
```bash
git clone https://github.com/oss-bit/Facial_Recognition.git
cd Facial_Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Loading Reference Images

To add reference faces to the system:

```python
from face_recognition import load_data_dir

# Load images from a directory
load_data_dir("path/to/reference/images")
```

### Running Real-time Recognition

To start the facial recognition system:

```python
python main.py
```

The system will:
1. Access your webcam
2. Detect faces in real-time
3. Compare detected faces against the reference database
4. Draw rectangles around recognized faces

## Configuration

Key parameters can be adjusted in the code:

```python
# Minimum face size threshold (in pixels)
proximity_threshold = 80

# Similarity threshold for face matching (lower = stricter matching)
similarity_threshold = 0.3
```

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade Classifier to detect faces in the video stream

2. **Feature Extraction**: 
   - Crops detected faces
   - Passes them through EfficientNet-B0
   - Extracts 2048-dimensional feature vectors

3. **Similarity Search**:
   - Uses ChromaDB to efficiently search for similar faces
   - Compares feature vectors using cosine similarity
   - Matches are determined based on the similarity threshold

4. **Visualization**:
   - Draws blue rectangles around recognized faces
   - Only processes faces above the proximity threshold

#

## Customization

### Using a Different Model

You can modify the feature extraction model by changing:

```python
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)
```

### Adjusting Face Detection

Modify the face detection parameters:

```python
faces = face_detector.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4)
```

## Contributing
Contributions to this project are warmly welcomed! Whether you're fixing bugs or adding features, your help is appreciated. Feel free to submit a pull request or open an issue to discuss potential improvements. 
## License

This project is licensed under the MIT License - see the LICENSE file for details.


