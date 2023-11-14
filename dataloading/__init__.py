from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, items: list, class_to_idx: dict, transforms=None):
        self.items = items
        self.transforms = transforms
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        label = self.class_to_idx[label]
        img = Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.items)


class SquareImage:
    def __call__(self, img):
        w, h = img.size
        max_ind = np.argmax([w, h])
        diff = np.abs(w - h)
        pad_pix = diff//2
        if diff % 2 == 0:
            pad1, pad2 = pad_pix, pad_pix
        else:
            pad1, pad2 = pad_pix, pad_pix + 1

        #  If a sequence of length 4 is provided to v2.Pad
        # this is the padding for the left, top, right and bottom borders
        if max_ind == 0:
            pad = [0, pad1, 0, pad2]
        else:
            pad = [pad1, 0, pad2, 0]
        padder = v2.Pad(padding=pad, padding_mode='reflect')
        return padder(img)
