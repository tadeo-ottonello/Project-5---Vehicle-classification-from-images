"""
This script will be used to separate and copy images coming from
`training_image_set.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...
"""
import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str -> Full path to raw images folder.

    labels : str -> Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """

    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)
    if not os.path.exists(os.path.join(output_data_folder, "test")):
        os.mkdir(os.path.join(output_data_folder, "test"))
    if not os.path.exists(os.path.join(output_data_folder, "train")):
        os.mkdir(os.path.join(output_data_folder, "train"))

    data = pd.read_csv(labels)
    data["class"] = data["class"].apply(lambda x: x.replace(" ", "_"))

    for row in data.itertuples():
        if not os.path.exists(os.path.join(output_data_folder, row[3], row[2])):
            os.mkdir(os.path.join(output_data_folder, row[3], row[2]))
        if not os.path.exists(os.path.join(output_data_folder, row[3], row[2], row[1])):
            os.link(os.path.join(data_folder, row[1]), os.path.join(output_data_folder, row[3], row[2], row[1]))


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
