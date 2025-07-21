"""
# dataset.py
This module defines the Dataset class for handling the histopathologic cancer detection dataset.
It includes methods for loading the dataset, validating paths, and retrieving image information.
It also defines the ImageInfo class to hold information about individual images.
"""

import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class ImageInfo:
    """Class to hold information about an image in the dataset.
    Attributes:
        image_path (str): Path to the image file.
        image_id (str): Unique identifier for the image.
        shape (tuple): Shape of the image (height, width, channels).
        image_label (int): Label of the image (0 for non-cancerous, 1 for cancerous).
    """

    def __init__(self, image_path: str, image_id: str, shape: tuple, image_label: int):
        self.image_path = image_path
        self.image_id = image_id
        self.shape = shape
        self.image_label = image_label

    def __repr__(self):
        return f"""ImageInfo(
            image_path={self.image_path}, 
            image_id={self.image_id}, 
            shape={self.shape}, 
            image_label={self.image_label})"""

    def __str__(self):
        return f"""Image ID: {self.image_id},
            Label: {self.image_label}, 
            Shape: {self.shape}, 
            Path: {self.image_path}"""

    def __eq__(self, other):
        if not isinstance(other, ImageInfo):
            return False
        return (
            self.image_id == other.image_id
            and self.image_path == other.image_path
            and self.shape == other.shape
            and self.image_label == other.image_label
        )


class Dataset:
    """Class to handle the dataset for histopathologic cancer detection.
    This class provides methods to load the dataset, validate paths, and retrieve image information.
    Attributes:
        folder_path (str): Path to the dataset folder.
        train_labels_file (str): Path to the training labels CSV file.
        train_images_dir (str): Directory containing training images.
        test_images_dir (str): Directory containing test images.
        sample_submission_file (str): Path to the sample submission CSV file.
        image_shape (tuple): Shape of the images in the dataset.
        image_size (tuple): Size of the images in the dataset.
        train_df (pd.DataFrame): DataFrame containing training data.
    """

    def __init__(self, folder_path):

        self._folder_path = folder_path
        self._train_labels_file = f"{folder_path}/train_labels.csv"
        self._train_images_dir = f"{folder_path}/train"
        self._test_images_dir = f"{folder_path}/test"
        self._sample_submission_file = f"{folder_path}/sample_submission.csv"

        self._validate_folder()

        self._image_shape = (0, 0, 0)
        self._image_size = (0, 0)

        self._train_df = self._load_train_data()

        self._set_shape_size()

    def __repr__(self) -> str:
        return (
            f"Dataset(folder_path={self._folder_path}, "
            f"train_labels_file={self._train_labels_file}, "
            f"train_images_dir={self._train_images_dir}, "
            f"test_images_dir={self._test_images_dir}, "
            f"sample_submission_file={self._sample_submission_file})"
        )

    @property
    def image_size(self):
        """Returns the size of the images in the dataset."""
        return self._image_size

    @property
    def image_shape(self):
        """Returns the shape of the images in the dataset."""
        return self._image_shape

    @property
    def train_labels_file(self):
        """Returns the path to the training labels file."""
        return self._train_labels_file

    @property
    def train_images_dir(self):
        """Returns the directory containing training images."""
        return self._train_images_dir

    @property
    def test_images_dir(self):
        """Returns the directory containing test images."""
        return self._test_images_dir

    @property
    def train_df(self):
        """Returns the DataFrame containing training data."""
        return self._train_df

    def get_random_train_image(self) -> ImageInfo:
        """Returns a random image ID from the training DataFrame."""
        image_id = random.choice(self._train_df["id"].tolist())
        print(f"Showing random image with ID: {image_id}")

        image_path = os.path.join(self._train_images_dir, image_id + ".tif")
        print(f"Showing random image from: {image_path}")
        shape = plt.imread(image_path).shape
        label = self._train_df[self._train_df["id"] == image_id]["label"].values[0]
        return ImageInfo(
            image_path=image_path, image_id=image_id, shape=shape, image_label=label
        )

    def load_tf_dataset(self) -> tf.data.Dataset:
        """
        Loads the TensorFlow dataset for training.
        This method reads the training labels and image paths, decodes the images,
        resizes them, and prepares a TensorFlow dataset for training.
        Returns:
            tf.data.Dataset: A TensorFlow dataset containing the training images and labels.
        """
        skip_ids = self.get_skip_image_ids()
        if skip_ids:
            print(f"Skipping {len(skip_ids)} invalid images.")
        filtered_df = self._train_df[~self._train_df["id"].isin(skip_ids)]
        paths = [
            os.path.join(self._train_images_dir, iid + ".tif")
            for iid in filtered_df["id"]
        ]
        labels = filtered_df["label"].values
        ids = filtered_df["id"].values

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels, ids))
        dataset = dataset.map(
            self._load_tiff, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return dataset

    def get_skip_image_ids(self) -> list[str]:
        """
        Retrieves a list of image IDs that should be skipped due to invalid formats.
        This method checks each image in the training images directory and identifies those that are not in TIFF format.
        Returns:
            list[str]: A list of image IDs (without the .tif extension) that are invalid.
        """
        ids = []
        folder_path = self._train_images_dir
        for fname in os.listdir(folder_path):
            image_path = os.path.join(folder_path, fname)
            with open(image_path, "rb") as fobj:
                header = fobj.read(4)
                is_tif = header == b"II*\x00" or header == b"MM\x00*"
            if not is_tif:
                print(f"Skipping image {fname} due to invalid format.")
                ids.append(fname[:-4])  # Remove .tif extension
        return ids

    def _set_shape_size(self) -> None:
        """
        Sets the image shape and size based on a random training image.
        This method retrieves a random image from the training set and sets the
        image shape and size attributes accordingly.
        Returns:
            Dataset: The current instance of the Dataset class with updated image shape and size.
        """
        img_info = self.get_random_train_image()

        self._image_shape = img_info.shape
        self._image_size = (img_info.shape[0], img_info.shape[1])

    def _load_train_data(self):
        train_df = pd.read_csv(self._train_labels_file)
        return train_df

    def _validate_folder(self):
        if not os.path.exists(self._folder_path):
            raise ValueError(f"Folder path {self._folder_path} does not exist.")
        if not os.path.isfile(self._train_labels_file):
            raise ValueError(
                f"Train labels file {self._train_labels_file} does not exist."
            )
        if not os.path.isdir(self._train_images_dir):
            raise ValueError(
                f"Train images directory {self._train_images_dir} does not exist."
            )
        if not os.path.isdir(self._test_images_dir):
            raise ValueError(
                f"Test images directory {self._test_images_dir} does not exist."
            )
        if not os.path.isfile(self._sample_submission_file):
            raise ValueError(
                f"Sample submission file {self._sample_submission_file} does not exist."
            )

    def _load_tiff(self, path, label, img_id):  # Updated to accept img_id
        def load_py(p):
            p = p.numpy().decode("utf-8")
            return plt.imread(p)

        img = tf.py_function(load_py, [path], tf.uint8)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img.set_shape(self._image_shape)  # type: ignore
        label = tf.cast(label, tf.int32)
        label.set_shape(())
        return img, label, img_id
