# Face-Recognition-using-opencv
This repository contains a Python script for real-time face detection using a webcam and OpenCV's Haar cascade classifier. The script captures video from the webcam, processes each frame to detect faces, and displays the frames with rectangles drawn around detected faces.

## Features:
  - **Real-Time Video Capture:** Captures video feed from the default webcam.
  - **Face Detection:** Utilizes OpenCV's Haar cascade classifier to detect faces in each frame.
  - **Grayscale Conversion:** Converts frames to grayscale for improved detection performance.
  - **Visualization:** Draws rectangles around detected faces and displays the video feed in real-time.
  - -**Interactive:** Runs until the user presses the 'q' key to exit.

## Prerequisites:

  - Python 3.x
  - OpenCV

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/real-time-face-detection.git
   cd real-time-face-detection
   ```

2. **Install the required dependencies**:
   ```bash
   pip install opencv-python
   ```

## Usage

1. **Run the script**:
   ```bash
   python face_detection.py
   ```

2. **Exit**: Press the 'q' key to stop the video capture and close the window.

