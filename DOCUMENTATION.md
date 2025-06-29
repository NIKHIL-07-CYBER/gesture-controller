# HandController

HandController is a Python application that uses your webcam and hand gestures to control system functions such as volume, screen brightness, and browser tab switching. It leverages OpenCV, MediaPipe, and other libraries to provide a real-time gesture-based controller for your computer.

## Features
- **Volume Control:** Adjust system volume by pinching with your right hand (thumb and index finger).
- **Brightness Control:** Adjust screen brightness by pinching with your left hand (thumb and middle finger).
- **Tab Switching:**
  - Switch to the next browser tab by showing all five fingers on your right hand.
  - Switch to the previous browser tab by showing two fingers on your left hand.

## Requirements
- Python 3.7+
- Webcam
- Windows OS (for volume and brightness control)

### Python Packages
- opencv-python
- mediapipe
- numpy
- pyautogui
- screen-brightness-control
- pycaw
- comtypes

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the application:**
   ```bash
   python main.py
   ```
2. **Allow webcam access** when prompted.
3. **Use gestures in front of your webcam:**

### Examples

#### 1. Adjust Volume (Right Hand)
- Pinch your thumb and index finger together to decrease volume.
- Spread them apart to increase volume.

#### 2. Adjust Brightness (Left Hand)
- Pinch your thumb and middle finger together to decrease brightness.
- Spread them apart to increase brightness.

#### 3. Switch Browser Tabs
- **Next Tab:** Show all five fingers (open palm) with your right hand.
- **Previous Tab:** Show two fingers (index and middle) with your left hand.

#### 4. Exit
- Press the `q` key to quit the application.

## Notes
- Make sure your webcam is connected and not used by another application.
- The application displays the current gesture and FPS on the video feed window.
- Works best in well-lit environments.

## Troubleshooting
- If you encounter errors related to audio or brightness control, ensure you are running on Windows and have the required permissions.
- For best results, keep your hand within the camera frame and avoid background clutter.

## License
MIT License
