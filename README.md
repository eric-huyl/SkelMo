# SkelMotion

SkelMotion 是一个基于 **MediaPipe** 的姿态识别项目，能够从视频、图像和实时流中识别全身关节角度。该项目利用 MediaPipe 的姿态检测功能来分析人体关节并实时或从静态输入中计算其角度。

## 功能

- 从视频、图像或实时流中检测人体关节。
- 提取并计算全身姿态的关节角度。
- 实时姿态检测和分析。

## 技术栈

- **Python** 3.x
- **MediaPipe** 用于姿态识别
- **OpenCV** 用于处理视频和图像输入
- **NumPy** 用于数据处理
- **Matplotlib**（可选）用于可视化关节角度数据

## 安装与设置

### 1. 克隆仓库

```bash
git clone https://github.com/eric-huyl/SkelMotion.git
cd SkelMotion
```

### 2. 创建并激活 conda 环境

```bash
conda create --name pose-recognition python=3.10
conda activate pose-recognition
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行项目（待办事项!!!）

- **视频中的姿态检测：**

```bash
python pose_recognition.py --video_path path_to_video.mp4
```

- **图像中的姿态检测：**

```bash
python pose_recognition.py --image_path path_to_image.jpg
```

- **实时姿态检测（使用摄像头）：**

```bash
python pose_recognition.py --live
```

## 许可证

MIT

---
