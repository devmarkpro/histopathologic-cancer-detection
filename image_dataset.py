import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Custom Dataset for loading histopathological images and their labels.
    Args:
        data_dir (str): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``.
        label_df (pd.DataFrame): DataFrame containing image filenames and their corresponding labels.
        data_type (str): Type of data to load, either "train" or "test".
    Returns:
        tuple: (image, label) where image is a transformed tensor and label is the corresponding
                label for the image.
    """

    def __init__(self, data_dir, transform, label_df, data_type="train"):
        images_path = os.path.join(data_dir, data_type)
        file_names = sorted(
            [f for f in os.listdir(images_path) if f.lower().endswith(".tif")]
        )

        self.items = []  # list of (full_path, label)
        for fname in file_names:
            key = os.path.splitext(fname)[0]
            if key in label_df.index:
                label = int(label_df.loc[key].values[0])
                self.items.append((os.path.join(images_path, fname), label))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return img, label
