import os


def delete_all_files_in_folder(folder_path):
    # 列出文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 如果是文件，则删除它
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")