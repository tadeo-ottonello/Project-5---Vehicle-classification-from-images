import os
import cv2
from utils.utils import walkdir
from utils.detection import get_vehicle_coordinates


"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """

    for dirpath, filename in walkdir(data_folder):

        img = cv2.imread(os.path.join(dirpath, filename))

        coord = get_vehicle_coordinates(img)

        crop_img = img[coord[1]:coord[3], coord[0]:coord[2]]

        new_path = os.path.join(output_data_folder, dirpath.split("/")[-2], dirpath.split("/")[-1])

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        if not os.path.exists(os.path.join(new_path, filename)):
            cv2.imwrite(os.path.join(new_path, filename), crop_img)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
