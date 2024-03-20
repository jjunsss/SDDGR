from ast import arg
from pyexpat import model
from sched import scheduler
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
from custom_prints import over_label_checker, check_components
from termcolor import colored
from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset

#* visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import cv2
from pycocotools.coco import COCO

def draw_boxes_on_image(image, image_id ,boxes, labels, title):
    """
    image: 이미지 배열 (Height, Width, Channels)
    targets: 바운딩 박스와 레이블이 있는 딕셔너리 리스트
    title: 이미지의 제목
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    # boxes = target['boxes'].cpu().numpy()
    # labels = target['labels'].cpu().numpy()
    # height = target["orig_size"][0].item()
    # width = target["orig_size"][1].item()
    height = 512
    width = 512
    for box, label in zip(boxes, labels):
        cx, cy, w, h = box  
        x = cx - w / 2
        y = cy - h / 2
        x, w = x * width, w * width
        y, h = y * height, h * height

        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, str(label), color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.savefig(f"./{title}_{image_id}.jpg")

from glob import glob
def load_image_from_coco(image_id):
    dataDir = "./test_dataset"
    # img_info = coco.loadImgs([image_id])[0]
    img = next((x for x in glob(os.path.join(dataDir, "*.jpg")) if int(os.path.basename(x).split(".")[0]) == int(image_id)), None)
    # image_path = os.path.join(dataDir, img)
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 포맷을 사용하므로 RG로 변환합니다.
    return image