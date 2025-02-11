import os
import numpy as np
import json


def delete_all_files_in_folder(folder_path):
    # 列出文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 如果是文件，则删除它
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


def load_numpy_images_from_folder(folder_path="tmp"):
    """
    读取指定文件夹内的所有 .npy 文件并将它们作为 NumPy 数组加载到列表中。

    参数:
    - folder_path: 文件夹路径，包含 .npy 文件。

    返回:
    - images: 一个列表，包含所有加载的 NumPy 数组。
    """
    images = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理 .npy 文件
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)

            # 加载 NumPy 数组文件
            image = np.load(file_path)
            images.append(image)
            print(f"加载了 {filename}")

    return images


def write_to_json(filename: str, data: dict):
    with open(filename, 'w') as f:
        json.dump(data, f)
        print(f"Saved to {filename}")


def read_from_json(filename: str):
    with open(filename, 'r') as f:
        print(f"Loaded from {filename}")
        return json.load(f)
