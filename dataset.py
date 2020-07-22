from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import glob

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.stat()['entries'])

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            _, img_bytes = [next(txn.cursor()) for _ in range(index)][0]

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class CatDataset(Dataset):
    def __init__(self, path, categories, transform, resolution=8):
        def files_list_from_categories(categories):
            categories = set(categories.split(":"))
            files = []
            for filepath in glob.iglob(path+"/*"):
                cls_name = filepath.split("_")[-2]
                if cls_name in categories:
                    files.append(filepath)
            return None if len(files) == 0 else files

        self.image_paths = files_list_from_categories(categories)
        self.length = len(self.image_paths)
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img
