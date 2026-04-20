# Biometric Blink Detector

Real-time blink detection and gesture recognition system using MediaPipe and OpenCV, with live Arduino RGB LED feedback.

## Overview

This project uses a Mac webcam to detect eye blinks and hand gestures in real time. 
When a blink is detected, it triggers a color dance on an Arduino RGB LED. The system also tracks hand landmarks and responds to specific gestures.

## Features

- Real-time facial landmark detection using MediaPipe (468 landmarks)
- Eye Aspect Ratio (EAR) algorithm for accurate blink detection
- Hand gesture recognition (thumbs up, peace sign, open palm, fist)
- Arduino RGB LED reacts to blinks and gestures via serial communication
- HUD overlay with biometric scan aesthetic
- Video recording with audio muxing via ffmpeg

## Hardware

- MacBook (built-in webcam)
- Arduino Uno R3
- RGB LED (common cathode)
- USB-A to USB-C adapter

## Tech Stack

- Python 3.14
- MediaPipe (face landmarker + gesture recognizer)
- OpenCV
- tkinter
- pyserial
- Arduino (C++)

## Gesture Controls

| Gesture | Action |
|---|---|
| 👍 Thumbs up | LED solid red |
| ✌️ Peace sign | YAY popup + LED |
| ✊ Fist | Pause / resume |

## Setup

1. Clone the repo
2. Install dependencies: `pip install opencv-python mediapipe Pillow pyserial librosa`
3. Download MediaPipe models: `face_landmarker.task` and `gesture_recognizer.task`
4. Upload `arduino_sketch.cpp` to your Arduino Uno
5. Update the serial port in `cv_arduino.py` to match your device
6. Run: `python cv_arduino.py`

## Background

Built as a side project connecting computational neuroscience concepts (neural signal detection, closed-loop feedback) to physical hardware. 
The EAR blink detection mirrors the kind of biosignal decoding used in brain-computer interface research.
