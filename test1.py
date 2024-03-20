from pycocotools.coco import COCO
import os
import json

# COCO 데이터셋의 annotation 파일 경로를 지정합니다.
dataDir = './'
dataType = 'Total_coco.json'  # 'train2017' 또는 'val2017' 중 하나를 선택합니다.
annFile = "Total_coco.json"

# JSON 파일을 불러옵니다.
with open(annFile, 'r') as f:
    coco_data = json.load(f)
    

# 특정 이미지 ID를 지정합니다.
image_id = 234700  # 원하는 이미지 ID로 변경하세요.

# 이미지 ID에 해당하는 객체를 찾습니다.
for ann in coco_data['annotations']:
    if ann['image_id'] == image_id:
        # 객체의 카테고리 ID를 찾습니다.
        cat_id = ann['category_id']
        
        # 카테고리 ID에 해당하는 카테고리 이름을 찾습니다.
        for cat in coco_data['categories']:
            if cat['id'] == cat_id:
                print("Object: ", cat['name'])
                break