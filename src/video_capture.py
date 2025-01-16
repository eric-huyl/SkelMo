import cv2
import time
import numpy as np

# 函数：打开摄像头
def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    return cap

# 函数：保存图像为 NumPy 格式
def save_frame_as_numpy(frame, timestamp):
    filename = f"tmp/frame_{int(timestamp)}.npy"
    np.save(filename, frame)  # 使用 numpy 保存数组
    print(f"输出图像：{filename}")

# 函数：显示视频流
def show_video_stream(cap, func = save_frame_as_numpy):
    prev_time = time.time()  # 初始化时间
    while True:
        ret, frame = cap.read()  # 捕获每一帧
        if not ret:
            print("无法读取视频帧")
            break

        current_time = time.time()  # 获取当前时间

        # 每秒保存一张图像
        if current_time - prev_time >= 1:
            # func(frame, current_time)  # 保存当前帧
            prev_time = current_time  # 更新上次保存图像的时间

        # 显示视频流
        cv2.imshow('Live Video', frame)

        # 如果按下 'q' 键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

# 主函数
def main():
    # 打开摄像头
    cap = open_camera(0)

    # 显示视频流并保存每秒一张图像
    show_video_stream(cap)

if __name__ == "__main__":
    main()
