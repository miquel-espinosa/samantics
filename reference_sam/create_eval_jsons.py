import numpy as np
import argparse
import json
import random
from pycocotools.coco import COCO
import pylab
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import skimage.io as io
from concurrent.futures import ThreadPoolExecutor, as_completed

import socket
import torch
import torchvision.ops.boxes as bops

# List of multi-instance annotations that we will not consider.
# Note that annotating multiple instances in the same annotation should not be allowed in COCO
MULTI_INSTANCE_ANNOTATIONS = [50138,1549254,1049944,1549217,1048117]

def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Command line arguments.')

    # Add arguments
    parser.add_argument('--n_ref',
                        type=int,
                        required=True,
                        help='Number of reference images. Follows a cumulative strategy. \
                                E.g. 1-shot=[x_1], 2-shot=[x_1, x_2], 3-shot=[x_1, x_2, x_3], ...')

    # Criterions for selecting the images
    parser.add_argument('--min_area',
                        type=int,
                        default=32**2,
                        help='Minimum annotation area in px')

    parser.add_argument('--max_area',
                        type=int,
                        default=50,
                        help='Maximum annotation area in percentage of the image area. Range from 0 to 100.')
    
    parser.add_argument('--aspect_ratio_range',
                        type=list,
                        default=[1/5, 5],
                        help='Aspect ratio of image')
    
    parser.add_argument('--bbox_inside_frame',
                        type=int,
                        default=10,
                        help='Discard boxes found within n px of the edges. Default is 5px frame.')
    
    parser.add_argument('--segm_inside_frame',
                        type=int,
                        default=15,
                        help='Define the width of inner frame in px. Used for the segm_overlap_with_frame check.')
    
    parser.add_argument('--segm_overlap_with_frame',
                        type=int,
                        default=5,
                        help='Percentage of the segm_frame that is intersected by the segmentation mask.')
    
    parser.add_argument('--max_iou_segm',
                        type=float,
                        default=0.2,
                        help='Reference segm should not overlap with other segms above this threshold.')
    
    parser.add_argument('--max_iou_bbox',
                        type=float,
                        default=0.35,
                        help='Reference bbox should not overlap with other bboxes above this threshold.')
    
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    
    parser.add_argument('--save_plots',
                        action='store_true',
                        help='Save plots of the selected images')
    
    parser.add_argument('--varied_sizes',
                        action='store_true',
                        help='Include varied sizes for the ref images')

    parser.add_argument('--select',
                        type=str,
                        default='random',
                        help='How to select the n_ref images. Options: largest, random')

    parser.add_argument('--sortby',
                        type=str,
                        default='area',
                        help='Sort the images by some criterion. Options: area, random')
    
    parser.add_argument('--root',
                        type=str,
                        default='/localdisk/data1/Data/COCO',
                        help='Root directory')
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    # HACK: Manually set some of the arguments
    # Engineering server
    if socket.gethostname().endswith('.see.ed.ac.uk'):
        args.ann_path = f'{args.root}/annotations_trainval2017/annotations/instances_train2017.json'
        custom_path = '/localdisk/data2/Users/s2254242/projects_storage/ref_sam/eval_data'
        args.save_path = f'{custom_path}/{args.n_ref}shot_{args.seed}seed_select{args.select}_sortby{args.sortby}'
    # Orchid server
    elif socket.gethostname().endswith('.jc.rl.ac.uk'):
        args.root = f'/work/scratch-nopw2/mespi/sam-hq-hf/sam-hq-training/data/coco'
        args.ann_path = f'{args.root}/annotations/instances_train2017.json'
        args.save_path = f'{args.root}/refsam_eval/{args.n_ref}shot_{args.seed}seed_select{args.select}_sortby{args.sortby}'
        os.makedirs(f'{args.root}/refsam_eval', exist_ok=True)

    return args

def calculate_iou_segm(coco, gt_ann_id, pred_ann_id):
    gt_mask = coco.annToMask(gt_ann_id)
    pred_mask = coco.annToMask(pred_ann_id)
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask) > 0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou

def check_iou_segm(coco, annotation, max_iou_segm=0.2):
    all_ann_ids = coco.getAnnIds(imgIds=annotation['image_id'])
    all_ann_ids.remove(annotation['id'])
    image_annotations = coco.loadAnns(all_ann_ids)
    ious = [calculate_iou_segm(coco, annotation, ann_id) for ann_id in image_annotations]
    # print(ious)
    if any(iou > max_iou_segm for iou in ious):
        return False
    return True

def convert_bbox_coco_to_x1x2y1y2(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def check_iou_bbox(coco, annotation, max_iou_bbox=0.2):
    all_ann_ids = coco.getAnnIds(imgIds=annotation['image_id'])
    all_ann_ids.remove(annotation['id'])
    if len(all_ann_ids) == 0:
        return True
    boxes1_coco = [annotation['bbox']]
    boxes2_coco = [coco.loadAnns(ann_id)[0]['bbox'] for ann_id in all_ann_ids]
    boxes1 = torch.tensor([convert_bbox_coco_to_x1x2y1y2(box1) for box1 in boxes1_coco])
    boxes2 = torch.tensor([convert_bbox_coco_to_x1x2y1y2(box2) for box2 in boxes2_coco])
    ious = bops.box_iou(boxes1, boxes2)
    # print(ious)
    if ious.max() > max_iou_bbox:
        return False
    return True

def check_bbox_edges(coco, annotation, thres=5):
    img = coco.loadImgs(annotation['image_id'])[0] # Get image
    height, width = img['height'], img['width']
    bbox = convert_bbox_coco_to_x1x2y1y2(annotation['bbox']) # [x1, y1, x2, y2]
    if bbox[0] < thres or bbox[1] < thres \
        or bbox[2] > width-thres or bbox[3] > height-thres:
        return False
    return True

def get_area_overlap(coco, annotation, px=5):
    img = coco.loadImgs(annotation['image_id'])[0]  # Get image
    height, width = img['height'], img['width']
    
    segmentation_mask = coco.annToMask(annotation) # Create the binary mask for the segmentation
    
    # Create the binary mask for the frame
    frame_mask = np.zeros((height, width), dtype=np.uint8)
    frame_mask[:px, :] = 1  # Top frame
    frame_mask[-px:, :] = 1  # Bottom frame
    frame_mask[:, :px] = 1  # Left frame
    frame_mask[:, -px:] = 1  # Right frame
    
    # Calculate the intersection of the segmentation mask and the frame mask
    intersection_mask = np.logical_and(segmentation_mask, frame_mask)
    
    # Compute the area of the intersection
    intersection_area = np.sum(intersection_mask)
    
    # Compute percentage of the 5px frame that is intersected by the segmentation mask
    percentage = (intersection_area /  np.sum(frame_mask)) * 100
    
    return intersection_area, percentage

def check_segm_edges(coco, annotation, thres=5, percentage_thres=5):
    area_overlap, percentage = get_area_overlap(coco, annotation, thres)
    # print(area_overlap, percentage)
    return percentage < percentage_thres

# Multi-threaded version
def process_annotation(coco, annotation):
    return get_area_overlap(coco, annotation, px=5)
# Multi-threaded version
def compute_area_overlaps_multithreaded(coco, annotations, num_threads=multiprocessing.cpu_count()):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Using list comprehension with tqdm for progress bar
        results = list(tqdm(executor.map(process_annotation, coco, annotations), total=len(annotations)))
    return results

def process_images_for_category(cat_name, imgs, anns, nref, args):
    # Create a figure with nref subplots arranged in two rows and five columns
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'{cat_name.upper()} - {nref} reference images - {args.seed} seed', fontsize=16, weight='bold')
    # Iterate over images and annotations
    for j, (img, ann) in enumerate(zip(imgs, anns), 1):  # Start enumeration from 1
        # Read the image from the URL
        I = io.imread(img['coco_url'])
        
        # Determine the subplot indices for the current iteration
        row = (j - 1) // 5  # Subtract 1 to align with zero-based indexing
        col = (j - 1) % 5   # Subtract 1 to align with zero-based indexing
        
        # Plot the image in the corresponding subplot
        axes[row, col].imshow(I)
        axes[row, col].axis('off')
        
        # Plot segmentation masks on the image
        for segmentation in ann['segmentation']:
            poly = np.array(segmentation).reshape((int(len(segmentation) / 2), 2))
            mask = Polygon(poly, facecolor='coral', edgecolor='red', linewidth=2, alpha=0.4)
            axes[row, col].add_patch(mask)
        
        # Add text annotation for image number
        axes[row, col].set_title(str(f"({j}) Image {ann['image_id']}\n   Ann {ann['id']}"), color='black', fontsize=11, weight='bold')

    # Adjust layout to prevent overlap of subplots
    fig.tight_layout()
    # Ensure the save path exists
    os.makedirs(f'{args.save_path}_images', exist_ok=True)
    fig.savefig(f'{args.save_path}_images/{cat_name}.png')
    plt.close(fig)

def plot_nref_images_with_bbox_and_segm(coco, category_dict, images, annotations, args):
    nref = args.n_ref
    # Safety check
    assert len(images) == len(annotations)
    assert len(category_dict) * nref == len(images)
    
    # Group images and annotations by category
    image_groups = [images[i:i + nref] for i in range(0, len(images), nref)]
    annotation_groups = [annotations[i:i + nref] for i in range(0, len(annotations), nref)]
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for imgs, anns in zip(image_groups, annotation_groups):
            cat_name = category_dict[anns[0]['category_id']]
            futures.append(executor.submit(process_images_for_category, cat_name, imgs, anns, nref, args))
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # If any exceptions occurred, they will be raised here


def main(args):
    
    print("Loading data ...")
    # Load JSON file
    with open(args.ann_path, 'r') as f:
        data = json.load(f)
    coco=COCO(args.ann_path)
    print("Data loaded!")
    
    # Access annotations
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    category_dict = { c['id']:c['name'] for c in categories}

    if args.select == 'largest':
        # Sort annotations by area. Largest to smallest
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
    elif args.select == 'random':
        # Shuffle with specified seed
        random.Random(args.seed).shuffle(annotations)
    else:
        raise ValueError(f"Invalid selection method: {args.select}")

    # Initialize a dictionary to store selected annotations
    selected_annotations = {category['id']: [] for category in categories}
    selected_images = {category['id']: [] for category in categories}

    # Create useful lists
    already_selected_imgs = []
    images_dict = {image['id']:image for image in images} # find image by its id
    
    # Select n_ref distinct annotations for each category
    # Use n_ref * len(categories) distinct images
    for annotation in tqdm(annotations):
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        img_h, img_w = images_dict[image_id]['height'], images_dict[image_id]['width']
        
        """ FILTER conditions:
            - image not already selected
            - annotation not crowd
            - annotation area > min_area (in pixels)
            - annotation area < max_area (in percentage)
            - image aspect ratio (width/height) within specified range
            - if (bbox not close to the image edges) or
                 (the segmentation mask area overlap with the 5px frame is smaller than 5%)
            - below iou_bbox overlap
            - below iou_segm overlap -> this one doesn't make sense, as it never happens 
            - and annotation is not a wrongly annotated multi-instance annotation
        """
        
                    # and (check_bbox_edges(coco, annotation, thres=args.bbox_inside_frame)) \
        if (len(selected_annotations[category_id]) < args.n_ref) \
                    and (image_id not in already_selected_imgs) \
                    and annotation['iscrowd'] == 0 \
                    and annotation['area'] > args.min_area \
                    and annotation['area'] < (img_h*img_w*args.max_area/100) \
                    and (img_w/img_h) >= args.aspect_ratio_range[0] \
                    and (img_w/img_h) <= args.aspect_ratio_range[1] \
                    and check_bbox_edges(coco, annotation, thres=args.bbox_inside_frame) \
                    and check_segm_edges(coco, annotation, thres=args.segm_inside_frame,
                                            percentage_thres=args.segm_overlap_with_frame) \
                    and check_iou_bbox(coco, annotation, max_iou_bbox=args.max_iou_bbox) \
                    and annotation['id'] not in MULTI_INSTANCE_ANNOTATIONS:
            selected_annotations[category_id].append(annotation)
            selected_images[category_id].append(images_dict[image_id])
            already_selected_imgs.append(image_id)
            
        # Check if we have selected enough annotations
        if all(len(annotations) == args.n_ref for annotations in selected_annotations.values()):
            break

    # Within each category, sort annotations by area, large to small. Sort images with the same order
    if args.sortby == 'area':
        for c in selected_annotations:
            selected_annotations[c], selected_images[c] = (list(t)
                                                                for t in zip(*sorted(
                                                                            zip(selected_annotations[c],
                                                                                selected_images[c]),
                                                                                key=lambda x: x[0]['area'],
                                                                                reverse=True)))
    
    
    # CHECKS that all the images are different (even for distinct categories)
    ann_image_ids = [ann['image_id'] for c in selected_annotations for ann in selected_annotations[c]]
    image_ids = [image['id'] for c in selected_images for image in selected_images[c]]
    assert len(ann_image_ids) == len(np.unique(ann_image_ids))
    assert len(image_ids) == len(np.unique(image_ids))
    for c in selected_annotations:
        ids_class = [ann['id'] for ann in selected_annotations[c]]
        assert len(np.unique(ids_class)) == len(ids_class)
    # CHECKS imgs and anns correspond
    for img_id, ann_img_id in zip(image_ids,ann_image_ids):
        assert img_id == ann_img_id
    
    # dict to list
    annotations_list = [annotation for annotations in selected_annotations.values() for annotation in annotations]
    images_list = [image for images in selected_images.values() for image in images]
    # Create a new JSON object following the same format as the original COCO annotations file
    selected_data = {
        "info": data['info'],
        "licenses": data['licenses'],
        "images": images_list,
        "annotations": annotations_list,
        "categories": data['categories']
    }
    
    # Save JSON with unique name
    selected_annotations_path = f'{args.save_path}.json'
    with open(selected_annotations_path, 'w') as f:
        json.dump(selected_data, f)
        
    if args.save_plots:
        # Check if images dir exists otherwise create
        if not os.path.exists(f'{args.save_path}_images'):
            os.makedirs(f'{args.save_path}_images')
        # Plot n_ref images with bbox and segm
        plot_nref_images_with_bbox_and_segm(coco, category_dict, images_list, annotations_list, args)


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Run the main function
    main(args)