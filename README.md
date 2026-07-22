# Driver Drowsiness Detection System

## About the Project

This project is a Driver Drowsiness Detection System developed using Python, OpenCV and MediaPipe. It monitors the driver's face through a webcam and detects signs of drowsiness based on eye closure, yawning and head movement. When drowsiness is detected, the system plays an alarm to alert the driver.

This project was developed as a college mini project to understand the practical use of Computer Vision and facial landmark detection.

---

## Features

- Detects eye closure using Eye Aspect Ratio (EAR)
- Detects yawning using Mouth Aspect Ratio (MAR)
- Detects head tilt
- Plays an alarm when drowsiness is detected
- Displays real-time webcam output

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pygame

---

## Project Structure

```
driver-drowsiness-detection/
│
├── src/
│   ├── drowsiness_detector.py
│   └── generate_alarm.py
│
├── sounds/
│   └── alarm.wav
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

1. Install the required libraries

```
pip install -r requirements.txt
```

2. Generate the alarm sound

```
python src/generate_alarm.py
```

3. Run the project

```
python src/drowsiness_detector.py
```

---

## Author

Vikram Singh

B.Tech Artificial Intelligence & Machine Learning
