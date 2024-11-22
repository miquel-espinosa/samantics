import json
import random
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py <train_annotations_path> <selected_annotations_path>")
    sys.exit(1)
# Path for annotations
train_annotations_path = sys.argv[1]
# train_annotations_path = "/localdisk/data1/Data/COCO/annotations_trainval2017/annotations/instances_train2017.json"
# Save the selected annotations to a new JSON file
selected_annotations_path = sys.argv[2]
# selected_annotations_path = f"/localdisk/data2/Users/s2254242/datasets/coco/precompute_{n_ref}_shot_annotations_ref_train.json"

# Number of reference images per category
n_ref = 10

# Load JSON file
with open(train_annotations_path, 'r') as f:
    data = json.load(f)

# Access annotations
images = data['images']
annotations = data['annotations']
categories = data['categories']

# Initialize a dictionary to store selected annotations
selected_annotations = {category['id']: [] for category in categories}
selected_images = {category['id']: [] for category in categories}

# (Optional) shuffle the original annotations to ensure randomness
# random.shuffle(original_annotations)

already_selected_imgs = []
images_dict = {image['id']:image for image in images} # find image by its id


# Select n_ref distinct annotations for each category
# Use n_ref * len(categories) distinct images
for annotation in annotations:
    category_id = annotation['category_id']
    image_id = annotation['image_id']
    if (len(selected_annotations[category_id]) < n_ref) and (image_id not in already_selected_imgs):
        selected_annotations[category_id].append(annotation)
        selected_images[category_id].append(images_dict[image_id])
        already_selected_imgs.append(image_id)

    # Check if we have selected 5 annotations for each category
    if all(len(annotations) == 5 for annotations in selected_annotations.values()):
        break


# Check that all the images are different (even for distinct categories)
ann_image_ids = [ann['image_id'] for c in selected_annotations for ann in selected_annotations[c]]
image_ids = [image['id'] for c in selected_images for image in selected_images[c]]
print(f"Total num of ann_ids: {len(ann_image_ids)}, Total num of unique ann_ids: {len(np.unique(ann_image_ids))}")
print(f"Total num of img_ids: {len(image_ids)}, Total num of unique img_ids: {len(np.unique(image_ids))}")
for c in selected_annotations:
    ids_class = [ann['id'] for ann in selected_annotations[c]]
    assert len(np.unique(ids_class)) == len(ids_class)
# assert len(ids) == len(np.unique(ids))

# Check imgs and anns correspond
for img_id, ann_img_id in zip(image_ids,ann_image_ids):
    assert img_id == ann_img_id

# Create a new JSON object following the same format as the original COCO annotations file
selected_data = {
    "info": data['info'],
    "licenses": data['licenses'],
    "images": [image for images in selected_images.values() for image in images], # dict to list
    "annotations": [annotation for annotations in selected_annotations.values() for annotation in annotations], # dict to list
    "categories": data['categories']
}

with open(selected_annotations_path, 'w') as f:
    json.dump(selected_data, f)