import json, os
import random  
from tqdm import tqdm
threshold = 50

data_path = "/data/jjunsss/7010set/T1/7010_T1B_200L_8~4conf/"
# Load JSON data
json_dir = os.path.join(data_path, "annotations/pseudo_data.json")
img_dir = os.path.join(data_path, "images")
with open(json_dir, 'r') as f:
    data = json.load(f)

# Initialize counts and mappings
category_counts = {}
image_categories = {}

# Calculate counts and record categories per image
for annotation in data['annotations']:
    category_id = annotation['category_id']
    image_id = annotation['image_id']
    category_counts[category_id] = category_counts.get(category_id, 0) + 1
    image_categories.setdefault(image_id, list()).append(category_id)

image_ids = list(image_categories.keys())
random.shuffle(image_ids)  # 이미지 ID를 랜덤하게 섞습니다.

images_to_remove = set()
for image_id in image_ids:
    categories = image_categories[image_id]
    if all(category_counts[cat_id] > threshold for cat_id in categories):
        images_to_remove.add(image_id)
        for cat_id in categories:
            category_counts[cat_id] -= 1

# Filter annotations
annotations_to_keep = [anno for anno in data['annotations'] if anno['image_id'] not in images_to_remove]
data['annotations'] = annotations_to_keep

# Remove image files
for image_id in tqdm(images_to_remove, desc="Deleting images"):
    image_info = next((img for img in data['images'] if img['id'] == image_id), None)
    if image_info:
        image_file_path = os.path.join(img_dir, image_info['file_name'])
        os.remove(image_file_path)

print(f"Processed and saved the filtered data to 'pseudo_data.json'.")

# Filter images
image_ids_to_keep = {anno['image_id'] for anno in annotations_to_keep}
data['images'] = [img for img in data['images'] if img['id'] in image_ids_to_keep]

# Write updated JSON data
with open(json_dir, 'w') as f:
    json.dump(data, f, indent=4)



# Load JSON data
json_dir = os.path.join(data_path, "annotations/pseudo_data.json")

with open(json_dir, 'r') as f:
    data = json.load(f)

# 카테고리별 어노테이션 개수를 저장할 딕셔너리 초기화
category_counts = {}

# 각 어노테이션을 순회하며 카테고리별로 개수 세기
for annotation in data['annotations']:
    category_id = annotation['category_id']
    if category_id not in category_counts:
        category_counts[category_id] = 0
    category_counts[category_id] += 1

# 카테고리 ID와 이름을 매핑하기 위한 딕셔너리 생성
category_id_to_name = {}
for category in data['categories']:
    category_id_to_name[category['id']] = category['name']

sorted_category_id_to_name = dict(sorted(category_counts.items(), key=lambda item: item[0]))

# 결과 출력
for category_id, count in sorted_category_id_to_name.items():
    print(f"ID: {category_id} Category: {category_id_to_name[category_id]}, Count: {count}")
    
# Get the list of image files that are referenced in the JSON data
referenced_images = {img['file_name'] for img in data['images']}

# Get all image files from the image directory
all_images = set(os.listdir(img_dir))

# Identify images not referenced in the JSON data
unreferenced_images = all_images - referenced_images

# Remove the unreferenced image files
for img_file in unreferenced_images:
    img_path = os.path.join(img_dir, img_file)
    os.remove(img_path)
    print(f"Removed unreferenced image: {img_path}")

print("Finished removing unreferenced images.")