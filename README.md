# Deadlift-counter-computer-vision

Real-time deadlift repetition counter using computer vision and machine learning.

The **Deadlift Counter** is a real-time fitness application that automatically counts deadlift repetitions using **computer vision** and **machine learning**. It uses a webcam to detect human pose landmarks, classifies the deadlift movement into **up** and **down** stages, and counts repetitions based on movement transitions.

------------------------------------------------------------------
## Features
- Real-time deadlift repetition counting
- Webcam-based human pose detection
- Stage classification (up / down)
- Confidence score display for predictions
- Interactive GUI with live video feed
- Reset repetition counter functionality

------------------------------------------------------------------
## Technologies Used
- **Python**
- **MediaPipe** – Pose estimation
- **OpenCV** – Video capture and processing
- **Machine Learning (Pickle model)** – Pose classification
- **Tkinter & CustomTkinter** – Graphical User Interface
- **NumPy & Pandas** – Data processing

------------------------------------------------------------------

## Project Structure
deadlift-counter-computer-vision/
│
├── deadlift_counter.py # Main application file
├── landmarks.py # Pose landmark feature definitions
├── deadlift.pkl # Trained machine learning model
└── README.md # Project documentation

## System Requirements
### Software Requirements
- Python 3.8 or above
- Required libraries


### Hardware Requirements
- Webcam
- Computer/Laptop
- Minimum 4 GB RAM recommended

------------------------------------------------------------------

## How It Works
1. The webcam captures live video frames.
2. MediaPipe detects body pose landmarks.
3. Landmark coordinates are converted into feature vectors.
4. A trained ML model classifies the pose as **up** or **down**.
5. A repetition is counted when a valid transition from **down → up** is detected.
6. The GUI updates the repetition count, stage, and confidence in real time.

------------------------------------------------------------------

## How to Run the Project
1. Clone or download this repository.
2. Install the required libraries
3. Make sure deadlift.pkl is in the same directory as the Python files.
4. Run the application:
   ython deadlift_counter.py
5. Stand in front of your webcam and perform deadlifts.
 






