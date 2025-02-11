import os
import pose

def read_files_in_order(folder_path, joint_angles):
    """
    Read files in a folder in order and add their content to the JointAngles instance.
    
    :param folder_path: The path to the folder containing the files
    :param joint_angles: An instance of the JointAngles class
    """
    files = sorted(os.listdir(folder_path))
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            data = file.read()
        

