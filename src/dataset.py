import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import math
import numpy as np
import sys
from utils import RandomMask, box_mask


class INP_Dataset(Dataset):
    def __init__(self, device, image_flist, mask_type, test_mask_flist=None, resolution=256):
        super().__init__()
        self.device = device
        self.image_list = self.load_flist(image_flist)
        self.mask_type = mask_type
        self.test_mask_list = self.load_flist(test_mask_flist)
        self.resolution = resolution

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image = Image.open(self.image_list[index]).resize((self.resolution, self.resolution))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = (image / 255.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).to(self.device)

        mask = self.get_mask(index)
        mask = torch.round(mask)

        return image, mask

    def load_name(self, index):
        name = self.image_list[index]
        return os.path.basename(name)

    def get_mask(self, index):

        if self.mask_type == 0:
            self.mask_type = 1 if np.random.uniform(0, 1) >= 0.1 else 2

        if self.mask_type == 1:
            mask = RandomMask(self.resolution)
            mask = torch.from_numpy(mask).float().to(self.device)
            mask = 1 - mask
            return mask

        if self.mask_type == 2:
            mask = box_mask([1, 3, self.resolution, self.resolution], 'cuda', 0.6, det=True).float()
            mask = 1 - mask
            return mask.squeeze(0)

        if self.mask_type == 3:
            mask = Image.open(self.test_mask_list[index % len(self.test_mask_list)]).resize(
                (self.resolution, self.resolution)).convert("L")
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).to(self.device).unsqueeze(0)
            return mask

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: images file path, images directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]

        return []


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def tensor_to_img(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().detach().permute(0, 2, 3, 1).float().numpy()
    image_result = numpy_to_pil(image)[0]
    return image_result


def make_flist(out_put_, img_path_):
    ext = {'.jpg', '.png', '.txt'}

    images = []
    for root, dirs, files in os.walk(img_path_):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1] in ext:
                images.append(os.path.join(root, file))

    images = sorted(images)
    np.savetxt(out_put_, images, fmt='%s')
