import torch
from pycocotools.coco import COCO
from datasets import build_dataset, get_coco_api_from_dataset

coco = COCO("/data/LG/real_dataset/total_dataset/didvepz/plustotal/output_json/train.json")
print(coco)


img_ids = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


imgs = coco.loadImgs(img_ids)


print(imgs)


# for img_id in img_ids:
#     anns = coco.getAnnIds(img_id)
#     load_anns = coco.loadAnns(ids=anns)
#     print(f'({img_id})id annotation load : {load_anns} \n')



build_dataset()
#test