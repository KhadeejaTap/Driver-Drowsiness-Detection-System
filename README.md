# Driver-Drowsiness-Detection-System
# Driver Drowsiness Detection System

A real-time computer vision tool that detects signs of driver fatigue using facial landmarks and head pose estimation. The system triggers an audio alert when drowsiness indicators such as prolonged eye closure or head tilting are detected.

## Features

- Real-time webcam monitoring
- Eye Aspect Ratio (EAR) calculation using MediaPipe Face Mesh
- Head pose estimation with Euler angles (pitch, yaw, roll)
- Audio alert system using threading to avoid blocking detection

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- playsound
- threading

## How to Run

1. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy playsound
