# Driver-Drowsiness-Detection-System

A real-time computer vision tool that detects signs of driver fatigue using facial landmarks and head pose estimation. The system triggers an audio alert when drowsiness indicators such as prolonged eye closure or head tilting are detected.

Demo: to be made

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
- simpleaudio
- threading

## How to Run

Clone the repository:
git clone https://github.com/KhadeejaTap/Driver-Drowsiness-Detection-System

Run Python 3.11 or slightly older, MediaPipe is not supported on newer versions
  
## Create a virtual environment:
- py -3.11 -m venv venv

## Activate it: 
- Mac/Linux: source venv/bin/activate
- Windows: venv\Scripts\activate


Install dependencies:
- pip install opencv-python mediapipe numpy simpleaudio


Run, Press q to quit

