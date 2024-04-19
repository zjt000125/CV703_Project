import json
import os
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from torchvision.utils import save_image

import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

from pycocotools import mask
from pycocotools.coco import COCO

from object_discovery.utils import (
    compact,
    rescale,
    slightly_off_center_crop,
    slightly_off_center_mask_crop,
    flatten_all_but_last,
)
from object_discovery.params import (
    merge_namespaces,
    training_params,
    slot_attention_params,
)

class CLEVRWithMasksDataset(Dataset):
    # Dataset details: https://github.com/deepmind/multi_object_datasets#clevr-with-masks
    def __init__(
        self,
        data_root: str,
        clevr_transforms: Callable,
        mask_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.mask_transforms = mask_transforms
        self.max_n_objects = max_n_objects
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"

        self.data = h5py.File(self.data_root, "r")
        if self.max_n_objects:
            if self.split == "train":
                num_objects_in_scene = np.sum(self.data["visibility"][:70_000], axis=1)
            elif self.split == "val":
                num_objects_in_scene = np.sum(
                    self.data["visibility"][70_001:85_000], axis=1
                )
            elif self.split == "test":
                num_objects_in_scene = np.sum(
                    self.data["visibility"][85_001:100_000], axis=1
                )
            else:
                raise NotImplementedError
            self.indices = (
                np.argwhere(num_objects_in_scene <= self.max_n_objects).flatten()
                + {"train": 0, "val": 70_001, "test": 85_001}[self.split]
            )

    def __getitem__(self, index: int):
        if self.max_n_objects:
            index_to_load = self.indices[index]
        else:
            index_to_load = index
        img = self.data["image"][index_to_load] # (0, 255) (240, 320, 3) numpy.ndarray
        if self.split == "train":
            mask = self.data["mask"][index_to_load] # (0, 255) (11, 240, 320, 1) numpy.ndarray
            vis = self.data["visibility"][index_to_load] # 11 numpy.ndarray
            return self.clevr_transforms(img), self.mask_transforms(mask), vis # torch.Size([3, 128, 128])[-1, 1], torch.Size([11, 128, 128, 1])[0, 255], (11,)numpy.ndarray [0, 1]
        else:
            mask = self.data["mask"][index_to_load] # (0, 255) (11, 240, 320, 1) numpy.ndarray
            vis = self.data["visibility"][index_to_load] # 11 numpy.ndarray
            # save_image(torch.tensor(img / 255.0).float().permute(2, 0, 1), "./image.png")
            # save_image(torch.tensor(mask / 255.0).float().permute(0, 3, 1, 2), "./mask.png")
            return self.clevr_transforms(img), self.mask_transforms(mask), vis

    def __len__(self):
        return len(self.indices if self.max_n_objects else self.data["image"])
    
resolution = slot_attention_params.resolution

def image_transforms(image):
    current_transforms = [
    # image has shape (H x W x C)
    transforms.ToTensor(),  # rescales to range [0.0, 1.0]
    # image has shape (C x H x W)
    ]
    current_transforms.append(
            transforms.Lambda(rescale)
        )  # rescale between -1 and 1
    current_transforms.extend(
        [
            transforms.Lambda(slightly_off_center_crop),
            transforms.Resize(resolution),
        ]
    )
    clevr_transforms = transforms.Compose(current_transforms)
    return clevr_transforms(image)

def mask_transforms(mask):
    # Based on https://github.com/deepmind/deepmind-research/blob/master/iodine/modules/data.py#L115
    # `mask` has shape [max_num_entities, height, width, channels]
    mask = torch.from_numpy(mask)
    mask = slightly_off_center_mask_crop(mask)
    mask = torch.permute(mask, [0, 3, 1, 2])
    # `mask` has shape [max_num_entities, channels, height, width]
    flat_mask, unflatten = flatten_all_but_last(mask, n_dims=3)
    resize = transforms.Resize(
        resolution, interpolation=transforms.InterpolationMode.NEAREST
    )
    flat_mask = resize.forward(flat_mask)
    mask = unflatten(flat_mask)
    # `mask` has shape [max_num_entities, channels, height, width]
    mask = torch.permute(mask, [0, 2, 3, 1])
    # `mask` has shape [max_num_entities, height, width, channels]
    return mask


class COCO2017(Dataset):
    NUM_CLASSES = 81
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 
 89, 90]
    
    assert(NUM_CLASSES) == len(set(CAT_LIST))

    def __init__(self, root, split='train', year='2017', image_size=224, mask_size=224, return_gt_in_train=False):
        super().__init__()
        ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
        self.img_dir = os.path.join(root, '{}{}'.format(split, year))
        if not os.path.isdir(self.img_dir):
            self.img_dir = os.path.join(root, "images", '{}{}'.format(split, year))
            assert os.path.isdir(self.img_dir)
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.return_gt_in_train = return_gt_in_train

        self.ids = list(self.coco.imgs.keys())
        
        self.train_transform = transforms.Compose([
                            transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
        
        self.val_transform_image = transforms.Compose([
                               transforms.Resize(size = image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(size = image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.val_transform_mask = transforms.Compose([transforms.Resize(size = mask_size, interpolation=transforms.InterpolationMode.NEAREST),
                               transforms.CenterCrop(size = mask_size),
                               transforms.PILToTensor()])
        self.image_size = image_size

    def __getitem__(self, index):
        img, mask_instance, mask_class, mask_ignore = self._make_img_gt_point_pair(index)

        if self.split == "train" and (self.return_gt_in_train is False):
            
            img = self.train_transform(img)
            
            return img
        elif self.split == "train" and (self.return_gt_in_train is True):
            img = self.val_transform_image(img)
            mask_class = self.val_transform_mask(mask_class)
            mask_instance = self.val_transform_mask(mask_instance)
            mask_ignore = self.val_transform_mask(mask_ignore)

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask_class = TF.hflip(mask_class)
                mask_instance = TF.hflip(mask_instance)
                mask_ignore = TF.hflip(mask_ignore)
            
            mask_class = mask_class.squeeze().long()
            mask_instance = mask_instance.squeeze().long()
            mask_ignore = mask_ignore.squeeze().long()

            return img, mask_instance, mask_class, mask_ignore        
        elif self.split =='val':

            img = self.val_transform_image(img)
            mask_class = self.val_transform_mask(mask_class).squeeze().long()
            mask_instance = self.val_transform_mask(mask_instance).squeeze().long()
            mask_ignore = self.val_transform_mask(mask_ignore).squeeze().long().unsqueeze(0)
            
            return img, mask_instance, mask_class, mask_ignore
        else:
            raise

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _targets = self._gen_seg_n_insta_masks(cocotarget, img_metadata['height'], img_metadata['width'])
        mask_class = Image.fromarray(_targets[0])
        mask_instance = Image.fromarray(_targets[1])
        mask_ignore = Image.fromarray(_targets[2])
        return _img, mask_instance, mask_class, mask_ignore

    def _gen_seg_n_insta_masks(self, target, h, w):
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        insta_mask = np.zeros((h, w), dtype=np.uint8)
        ignore_mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for i, instance in enumerate(target, 1):
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                seg_mask[:, :] += (seg_mask == 0) * (m * c)
                insta_mask[:, :] += (insta_mask == 0) * (m * i)
                ignore_mask[:, :] += m
            else:
                seg_mask[:, :] += (seg_mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
                insta_mask[:, :] += (insta_mask == 0) * (((np.sum(m, axis=2)) > 0) * i).astype(np.uint8)
                ignore_mask[:, :] += (((np.sum(m, axis=2)) > 0) * 1).astype(np.uint8)

        # Ignore overlaps
        ignore_mask = (ignore_mask>1).astype(np.uint8)

        all_masks = np.stack([seg_mask, insta_mask, ignore_mask])
        return all_masks

    def __len__(self):
        return len(self.ids)