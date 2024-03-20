# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO

class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1, ids_list = None, class_ids = None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys())) if ids_list == None else ids_list
        if not isinstance(class_ids, list):
            class_ids = [class_ids]
        self.class_ids = class_ids
        
        if class_ids is not None and ids_list == None:
            self.ids = []
            for c_idx in self.class_ids:
                img_ids = self.coco.getImgIds(catIds= c_idx)
                self.ids.extend(img_ids)
            self.ids = list(set(self.ids))
        
        #print(f"{dist.get_rank()} here image id list counts : {self.ids}")
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index): #! 여기서 Category Id를 순서대로 호출하면 될 것 같음
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index] #TODO : 기존에는 모든 존재하는 이미지 ID를 가져와서 사용하던 것을 변환 -> 미리 선언한 Class에 해당하는 이미지들만 선별
        
        if self.class_ids is not None: # 클래스 IDS에 해당하는 Target만 가져오는 것이 핵심
            target = [value for value in coco.loadAnns(coco.getAnnIds(img_id)) if value["category_id"] in self.class_ids]
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(int(img_id))[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
