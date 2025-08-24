import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Custom Dataset for loading histopathological images and their labels.
    - data_type in {"train", "test"}:
        * "train": returns (image_tensor, label:int)
        * "test" : returns (image_tensor, id:str)
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

    def __init__(self, data_dir, transform, label_df=None, data_type: str = "train"):
        assert data_type in {"train", "test"}, "data_type must be 'train' or 'test'"
        self.transform = transform
        self.data_type = data_type

        folder = "test" if data_type == "test" else "train"
        images_path = os.path.join(data_dir, folder)
        file_names = sorted(
            [f for f in os.listdir(images_path) if f.lower().endswith(".tif")]
        )

        self.ids = [os.path.splitext(f)[0] for f in file_names]
        full_paths = [os.path.join(images_path, f) for f in file_names]

        if data_type == "test":
            # No labels for test; keep ids for CSV submissions
            self.labels = None
            # For test we store (path, id) tuples
            self.items = list(zip(full_paths, self.ids))
        else:
            # Train/val mode: require labels and keep (path, label) to stay compatible
            if label_df is None:
                raise ValueError("label_df must be provided when data_type='train'.")

            items = []
            ids_kept, labels = [], []
            for p, id_ in zip(full_paths, self.ids):
                if id_ in label_df.index:
                    y = int(label_df.loc[id_].values[0])
                    items.append((p, y))
                    ids_kept.append(id_)
                    labels.append(y)
                # else: skip files without labels

            self.items = items  # <- (path, label) as before
            self.ids = ids_kept  # align with kept items
            self.labels = labels  # list[int]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, second = self.items[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        if self.data_type == "test":
            img_id = second  # (path, id)
            return img, img_id
        else:
            label = second  # (path, label)
            return img, label
