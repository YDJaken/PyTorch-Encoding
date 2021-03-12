import os
import torch
import numpy as np
from PIL import Image
from .base import BaseDataset


class arcs(BaseDataset):
    NUM_CLASS = 2
    def __init__(self, split="train", root = r"F:\色林错\dataSet", mode=None, transform=None,
                 target_transform=None, base_size=128, crop_size=128):
        super(arcs, self).__init__(root, split, mode,
                                   transform, target_transform, base_size, crop_size)
        self.NUM_CLASS = 2
        self.images, self.masks = get_arcs_pairs(self.root, self.split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError(
                "Found 0 images in subfolders of: " + self.root + "\n")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64')
        # target[target==0] = 1
        target[target==255] = 1
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)
    pass


def get_arcs_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            # if len(img_paths) > 50:
            #     break
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".tif"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.tif'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'train/data')
        mask_folder = os.path.join(folder, 'train/mask')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'val':
        img_folder = os.path.join(folder, 'validation/data')
        mask_folder = os.path.join(folder, 'validation/mask')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    return img_paths, mask_paths
