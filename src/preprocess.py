import os
from typing import List
from src.predict import keypoints_with_scores


def list_keypoints(images_folder: os.PathLike) -> List:
    keypoints_list = []

    images_name = os.listdir(images_folder)
    for name in images_name:
        img_path = os.path.join(images_folder, name)
        result = keypoints_with_scores(img_path)
        keypoints_list.append(result)

    return keypoints_list


def prepare_data(src: os.PathLike, dst: os.PathLike) -> None:
    """
    Generate the data for the pose classification model to training.

    Args:
        src: the path of the original stored pose image folder
        dst: the path of the processed training data folder
    """
    if not os.path.exists(src):
        raise ValueError("The 'src' path is not exists.")

    if not os.path.exists(dst):
        os.mkdir(dst, 644)

    dirs = os.listdir(src)
    pose_folders = []

    # find all the folders of the different pose classes
    for folder in dirs:
        folder_path = os.path.join(src, folder)
        if os.path.isfile(folder_path):
            continue

        pose_folders.append(folder)


    # detect the keypoints of pose images and record to the dictionary 
    pose_keypoints = {}
    for pose in pose_folders:
        result = list_keypoints(os.path.join(src, pose))
        pose_keypoints.update({f"{pose}": result})

    # TODO: create the csv files
    print(pose_keypoints)
    print("done")
