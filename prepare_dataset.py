import os
from PIL import Image
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import sys
import shutil


class DataPreprocessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
        self.df = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
        self.label_dict = dict(zip(self.df["id"], self.df["label"]))
        self.output_dir = output_dir
        if len(self.output_dir) == 0:
            self.output_dir = self.data_dir

    def run(self, dir: str, forced=False) -> str:
        """
        Run the preprocessing on the specified directory.
        :param dir: Directory to process (train or test).
        """
        if dir == "train":
            return self._process_train_data(forced)
        elif dir == "test":
            return self._process_test_data(forced)
        else:
            raise ValueError("Invalid directory specified. Use 'train' or 'test'.")

    def _process_train_data(self, forced=False) -> str:
        # Create output directories once
        processed_dir = os.path.join(self.output_dir, "train_processed")

        if os.path.exists(processed_dir):
            print(f"processed directory exists")
            if not forced:
                print("forced is False, return without any modification")
                return processed_dir
            else:
                print("Removing the current directory")
                shutil.rmtree(processed_dir)

        print("creating directories for processed images")
        os.makedirs(os.path.join(processed_dir, "1"), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, "0"), exist_ok=True)

        # Get list of all TIFF files (assuming flat directory)
        files = [f for f in os.listdir(self.train_dir) if f.endswith(".tif")]

        # Prepare arguments for parallel processing with tqdm
        tasks = []
        for file in tqdm(files, desc="Preparing tasks"):
            file_path = os.path.join(self.train_dir, file)
            id_str = file[:-4]
            label = self.label_dict.get(id_str)
            if label is not None:  # Skip if no label (edge case)
                new_file_path = os.path.join(processed_dir, str(label), id_str + ".jpg")
                tasks.append((file_path, new_file_path))

        # Parallel conversion
        num_processes = os.cpu_count()
        with Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(self._convert_tiff_to_jpeg_parallel, tasks),
                    total=len(tasks),
                    desc="Converting images",
                )
            )
        return processed_dir

    def _process_test_data(self, forced=False) -> str:
        # Create output directory once
        processed_dir = os.path.join(self.output_dir, "test_processed")

        if os.path.exists(processed_dir):
            print(f"processed directory exists")
            if not forced:
                print("forced is False, return without any modification")
                return processed_dir
            else:
                print("Removing the current directory")
                shutil.rmtree(processed_dir)

        print("creating directories for processed images")

        os.makedirs(processed_dir, exist_ok=True)

        # Get list of all TIFF files (assuming flat directory)
        files = [f for f in os.listdir(self.test_dir) if f.endswith(".tif")]

        # Prepare arguments for parallel processing with tqdm
        tasks = []
        for file in tqdm(files, desc="Preparing tasks"):
            file_path = os.path.join(self.test_dir, file)
            new_file_path = os.path.join(processed_dir, file[:-4] + ".jpg")
            tasks.append((file_path, new_file_path))

        # Parallel conversion
        num_processes = os.cpu_count()
        with Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(self._convert_tiff_to_jpeg_parallel, tasks),
                    total=len(tasks),
                    desc="Converting images",
                )
            )
        return processed_dir

    @staticmethod
    def _convert_tiff_to_jpeg_parallel(args):
        tiff_file, jpeg_file = args
        with Image.open(tiff_file) as img:
            img.convert("RGB").save(jpeg_file, "JPEG")


if __name__ == "__main__":

    data_dir = "./data/histopathologic-cancer-detection"
    # read the arg from the command line for train or test directory
    # if no arg is provided, process both train and test directories
    if len(sys.argv) > 1:
        dir_to_process = sys.argv[1]
    else:
        dir_to_process = "both"

    preprocessor = DataPreprocessor(data_dir, output_dir=data_dir)
    if dir_to_process == "train" or dir_to_process == "both":
        print("Processing training data...")
        o = preprocessor.run("train")
        print(f"Training data processed and saved to: {o}")
    elif dir_to_process == "test" or dir_to_process == "both":
        print("Processing test data...")
        o = preprocessor.run("test")
        print(f"Test data processed and saved to: {o}")
    else:
        print("Invalid argument. Use 'train', 'test', or 'both'.")
        sys.exit(1)
