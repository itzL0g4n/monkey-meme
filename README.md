# Gesture Recognition Mirror

This project implements a real-time computer vision application that detects hand gestures and facial proximity using MediaPipe and OpenCV. The system analyzes the user's webcam feed to identify specific gestures and displays a corresponding image state alongside the video stream.

## Features

The application currently supports the detection of the following states:
* **Neutral:** Default state when no specific gesture is recognized.
* **Thumbs Up:** Detects when the thumb is extended upward while other fingers are folded.
* **Pointing:** Detects when the index finger is extended while other fingers are folded.
* **Thinking:** Detects a "pointing" gesture where the index finger is positioned near the face area.

## Prerequisites

* Python 3.7 or higher.
* A functional webcam.
* The following image assets must be present in the same directory as the script:
    * `neutral.jpg`
    * `thumbsup.jpg`
    * `pointing.jpg`
    * `thinking.jpg`


## Run the following command to install the requirements:

```
pip install -r requirements.txt
```

## Usage

To start the application, execute the main script from your terminal:

```
python monkey.py
```

## Controls

* **q** or **ESC**: Press either key to close the application window and terminate the process.

## Technical Details

The system utilizes MediaPipe Hands to extract 21 hand landmarks and MediaPipe Face Detection to locate the facial bounding box. The logic calculates Euclidean distances between landmarks to determine finger folding states and checks the relative position of the index finger tip against the face coordinates to distinguish between "Pointing" and "Thinking" gestures.
