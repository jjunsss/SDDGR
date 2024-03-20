# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import util.misc as utils

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
from datasets.augmentation import *

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target

def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

class RandomAdjustSharpness(object):
    def __init__(self, *args, **kwargs):
        self.thr = 0.5
        self.random  = T.RandomAdjustSharpness(2, 0.5)
        
    def __call__(self, img, target):
        if random.random() > self.thr:
            return self.random(img), target
        else:
            return img, target

class ColorJitter(object):
    def __init__(self, *args, **kwargs):
        self.random  = T.ColorJitter(contrast= 0.1, brightness = 0.2)
        self.thr = 0.6
        
    def __call__(self, img, target):
        if random.random() > self.thr:
            return self.random(img), target
        else:
            return img, target

class RandomAugmetation(object):
    def __init__(self, N = 1):
        self.thr = 0.4
        self.N = N
        self.M = 1
      

    def fun(self, M):
        color_range = torch.arange(0, 0.9+1e-8, (0.9-0)/M).tolist()
        rotate_range = torch.arange(0, 30+1e-8, (30-0)/M).tolist()
        shear_range = torch.arange(0, 0.3+1e-8, (0.3-0)/M).tolist()
        translate_range = torch.arange(0, 250+1e-8, (250-0)/M).tolist()
        translate_bbox_range = torch.arange(0, 120+1e-8, (120-0)/M).tolist()
        
        Fun = {'Brightness' : Brightness, 'Color' : Color, 'Contrast' : Contrast,
        'Solarize' : Solarize, 'equalize' : Equalize, 'Sharpness' : Sharpness,
        'Posterize' : Posterize,
        }
        
        Mag = {'Brightness' : color_range, 'Color' : color_range, 'Contrast' : color_range, 
       'Solarize' : torch.arange(0, 256+1e-8, (256-0)/M).tolist()[::-1],
       'Rotate_BBox' : rotate_range, 'ShearX_BBox' : shear_range, 'ShearY_BBox' : shear_range,
       'TranslateX_BBox' : translate_range, 'TranslateY_BBox' : translate_range, 'Posterize' : torch.arange(4, 8+1e-8, (8-4)/M).tolist()[::-1],
       'SolarizeAdd' : torch.arange(0, 110+1e-8, (110-0)/M).tolist(), 'Sharpness' : color_range
        }
        key = list(Fun.keys())
        sample = np.random.choice(key, 1)
        
        # value 1 is represent to probability.
        if sample in ['Posterize', 'Solarize']:
            return Fun[sample[0]](1, Mag[sample[0]][M])        
        elif sample in ['equalize']:
            return Fun[sample[0]](1)  
        else:
            return Fun[sample[0]](1, Mag[sample[0]][M], minus = False)

    def __call__(self, img, target): #img : PIL image / target : dic
        boxes = target["boxes"]
        self.M = np.random.choice(range(1, 9), 1).item()
            
        augmentation = self.fun(self.M)
        label_tensor = target['labels']
        label_tensor = label_tensor.to('cpu')
        label_tensor_unique = torch.unique(label_tensor)
        limit2 = [28, 32, 35, 41, 56] 
        check_list2 = [idx.item() for idx in label_tensor_unique if idx.item() in limit2] #pz
        if random.random() > self.thr and len(check_list2) < 1: #and len(check_list2) < 1
            #print("\n augmentations: ", augmentation)
            aug_image, _ = augmentation(img, boxes)
            #target['boxes'] = aug_bboxes
            # if utils.is_main_process():     
            #     aug_image.save("/data/LG/real_dataset/total_dataset/test_dir/Deformable-DETR/datasets/chagned.jpg")
            #     img.save("/data/LG/real_dataset/total_dataset/test_dir/Deformable-DETR/datasets/img.jpg")
            return aug_image, target
        
        else:
            return img, target
        
def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        label_tensor = target['labels']
        label_tensor = label_tensor.to('cpu')
        label_tensor_unique = torch.unique(label_tensor)
        limit2 = [ 28, 32, 35, 41, 56]
        check_list2 = [idx.item() for idx in label_tensor_unique if idx.item() in limit2]
        
        if len(check_list2) < 1:
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])
            return crop(img, target, region)
        else:
            return img, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target=None):
        return F.to_tensor(img), target

class ToPIL(object):
    def __init__(self, ):
        self.ToPILImage = T.ToPILImage()
        
    def __call__(self, img, target):
        return self.ToPILImage(img), target
    
class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target
    
class Origin_Normalize(object):
    def __init__(self, ):
        pass
    
    def __call__(self, image, target=None):
        image = np.array(image)
        
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[:-1]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

class image_Normalize(object):
    def __init__(self, mean =[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
