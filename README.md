
---

# SkelMotion

SkelMotion is a pose recognition project based on **MediaPipe**, capable of identifying full-body joint angles from videos, images, and live streams. The project utilizes MediaPipe's pose detection capabilities to analyze human body joints and calculate their angles in real-time or from static inputs.

## Features

- Detect human body joints from video, image, or live stream.
- Extract and compute joint angles for full-body poses.
- Real-time pose detection and analysis.

## Tech Stack

- **Python** 3.x
- **MediaPipe** for pose recognition
- **OpenCV** for handling video and image inputs
- **NumPy** for data processing
- **Matplotlib** (optional) for visualizing joint angle data

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/eric-huyl/SkelMotion.git
cd SkelMotion
```

### 2. Create and activate the conda environment

```bash
conda create --name pose-recognition python=3.10
conda activate pose-recognition
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project(TODO!!!)

- **Pose detection on video:**

```bash
python pose_recognition.py --video_path path_to_video.mp4
```

- **Pose detection on image:**

```bash
python pose_recognition.py --image_path path_to_image.jpg
```

- **Real-time pose detection (using webcam):**

```bash
python pose_recognition.py --live
```


## License

MIT

---
