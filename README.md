# Industrial Robotics Model

USB Camera Feed with Real-time Object Detection for Raspberry Pi OS

## Overview

This project provides a Python script that captures video from a USB camera and performs real-time object detection using OpenCV's Deep Neural Network (DNN) module with the MobileNet SSD model. Detected objects are displayed with bounding boxes in a window.

## Features

- Real-time USB camera feed display
- Object detection using MobileNet SSD (lightweight model suitable for Raspberry Pi)
- Bounding boxes with class labels and confidence scores
- FPS counter display
- Automatic model download on first run
- Support for multiple camera indices

## Supported Objects

The model can detect 20 different object classes:
- People, vehicles (car, bus, bicycle, motorbike, aeroplane, boat, train)
- Animals (bird, cat, dog, cow, horse, sheep)
- Indoor objects (bottle, chair, diningtable, pottedplant, sofa, tvmonitor)

## Requirements

- Raspberry Pi (any model) running Raspberry Pi OS
- USB camera connected to the Raspberry Pi
- Python 3.7 or higher
- Internet connection (for first run to download model files)

## Installation

### 1. Update your system

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install system dependencies

```bash
sudo apt install -y python3-pip python3-opencv
```

### 3. Clone this repository

```bash
git clone https://github.com/ChenFangM/industrial-robotics-model.git
cd industrial-robotics-model
```

### 4. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

**Note:** On Raspberry Pi OS, you might need to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Connect your USB camera to the Raspberry Pi

2. Run the script:

```bash
python3 camera_object_detection.py
```

3. A window will open displaying the camera feed with detected objects highlighted by bounding boxes

4. Press `q` to quit the application

## Troubleshooting

### Camera not detected

- Check that your USB camera is properly connected
- Try running `ls /dev/video*` to see available camera devices
- The script automatically tries camera indices 0-4

### Low frame rate

- The frame rate depends on your Raspberry Pi model
- On older models, consider reducing the resolution in the script
- Close other applications to free up CPU resources

### Display issues over SSH

If running over SSH, you need X11 forwarding enabled:

```bash
ssh -X pi@raspberrypi
```

Or use VNC to access the Raspberry Pi desktop.

### Model download fails

If the automatic model download fails, manually download the files:

1. `deploy.prototxt` from: https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt
2. `mobilenet_iter_73000.caffemodel` from: https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel

Place both files in the same directory as `camera_object_detection.py`.

## License

See [LICENSE](LICENSE) file for details.