import os
import shutil
import random
import argparse
from tqdm import tqdm

def split_data_centralized(root_dir, output_dir, train_percent, val_percent, test_percent):
    """
    Recursively goes through all subfolders in the root directory,
    collects all files, and splits them into centralized 'train', 'val',
    and 'test' folders within the specified output directory based on the
    provided percentages. The original folder structure is flattened.

    Args:
        root_dir (str): The root directory containing the data.
        output_dir (str): The directory where the 'train', 'val', and 'test'
                           subfolders will be created.
        train_percent (float): Percentage of data for training (0.0 to 1.0).
        val_percent (float): Percentage of data for validation (0.0 to 1.0).
        test_percent (float): Percentage of data for testing (0.0 to 1.0).
    """
    if not (0 <= train_percent <= 1 and 0 <= val_percent <= 1 and 0 <= test_percent <= 1 and
            abs(train_percent + val_percent + test_percent - 1.0) < 1e-6):
        raise ValueError("Train, validation, and test percentages must sum to 1.0")

    all_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            all_files.append(os.path.join(subdir, file))

    random.shuffle(all_files)
    num_files = len(all_files)

    train_split_index = int(train_percent * num_files)
    val_split_index = train_split_index + int(val_percent * num_files)

    train_files = all_files[:train_split_index]
    val_files = all_files[train_split_index:val_split_index]
    test_files = all_files[val_split_index:]

    # Create output subfolders if they don't exist
    train_output_dir = os.path.join(output_dir, "train")
    val_output_dir = os.path.join(output_dir, "val")
    test_output_dir = os.path.join(output_dir, "test")

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Move files to their respective output folders with progress bars
    print("Moving files to train folder:")
    for src_path in tqdm(train_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(train_output_dir, filename)
        shutil.move(src_path, dst_path)

    print("Moving files to validation folder:")
    for src_path in tqdm(val_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(val_output_dir, filename)
        shutil.move(src_path, dst_path)

    print("Moving files to test folder:")
    for src_path in tqdm(test_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(test_output_dir, filename)
        shutil.move(src_path, dst_path)

    print("Data splitting complete into:", train_output_dir, val_output_dir, test_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data from a directory into centralized train, validation, and test sets.")
    parser.add_argument("root_dir", help="The root directory containing the data.")
    parser.add_argument("output_dir", help="The directory where the train, val, and test folders will be created.")
    parser.add_argument("--train", type=float, default=0.7, help="Percentage of data for training (default: 0.7).")
    parser.add_argument("--val", type=float, default=0.15, help="Percentage of data for validation (default: 0.15).")
    parser.add_argument("--test", type=float, default=0.15, help="Percentage of data for testing (default: 0.15).")

    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: Input directory '{args.root_dir}' does not exist.")
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    elif not os.path.isdir(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' is not a valid directory.")
    else:
        split_data_centralized(args.root_dir, args.output_dir, args.train, args.val, args.test)

#Example call :python dataset_partitioner.py /path/to/your/input/data /path/to/your/output/directory --train 0.8 --val 0.1 --test 0.1
