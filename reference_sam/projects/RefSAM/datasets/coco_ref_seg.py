import copy
import random

import lvis
import torch
import numpy as np

from mmengine.dataset import force_full_init
from mmengine.fileio import get_local_path
from mmengine.dataset.base_dataset import Compose
from mmengine.structures.instance_data import InstanceData

from mmdet.datasets.lvis import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class COCORefSegTrainDataset(CocoDataset):
    def __init__(
        self,
        with_ref=True,
        ref_equal_target=False,
        ignore_non_exclusive=False,
        max_sampling_try=10,
        category_ids_config=None,
        ref_pipeline=None,
        max_categories_training=99999,
        *args,
        **kwargs
    ):

        self.category_ids_config = category_ids_config

        # init will run the load_data_list function
        super(COCORefSegTrainDataset, self).__init__(*args, **kwargs)

        self.with_ref = with_ref
        self.ref_equal_target = ref_equal_target
        self.ignore_non_exclusive = ignore_non_exclusive
        self.max_sampling_try = max_sampling_try

        if self.ignore_non_exclusive:
            raise NotImplementedError

        if ref_pipeline is None:
            self.ref_pipeline = self.pipeline
        else:
            self.ref_pipeline = Compose(ref_pipeline)

        self.max_categories_training = max_categories_training


    def sample_ref_data(self, idx, target_img_id, gt_labels):
        # TODO: This function could potential have mismatches with the idx
        """
            Sample reference data for each category in gt_labels
        Args:
            idx: sample index
            target_img_id: id of the target image
            gt_labels: labels for the target image instances
        """
        exclude_labels = []
        ref_data = {}

        # Limit the number of categories to sample to avoid OOM
        if len(gt_labels) > self.max_categories_training:
            pre_include = np.random.permutation(len(gt_labels))[:self.max_categories_training]
        else:
            pre_include = None

        for i, label in enumerate(gt_labels):
            cat_id = self.cat_ids[label]
            cat_instances = self.catID_instances[cat_id]

            if len(cat_instances) == 0:
                exclude_labels.append(label)
                continue
            if pre_include is not None and i not in pre_include:
                exclude_labels.append(label)
                continue

            # Sample a random reference image that is not the target image.
            # Try for `max_sampling_try` times
            success = False
            for _ in range(self.max_sampling_try):
                sampled_ann = cat_instances[random.randint(0, len(cat_instances) - 1)]
                if self.coco.anns[sampled_ann]['image_id'] != target_img_id:
                    ref_data[label] = (self.coco.anns[sampled_ann]['image_id'], sampled_ann)
                    success = True
                    break
            if not success:
                exclude_labels.append(label)
        return ref_data, exclude_labels

    def prepare_ref_data(self, idx, image_id, instances):
        """
            Prepare reference data for the target image
        """
        # Unique class labels in the target image
        gt_labels = set([ins["bbox_label"] for ins in instances])
        # Sample 1 reference image for each class label
        ref_data, exclude_labels = self.sample_ref_data(idx, image_id, gt_labels)
        # ref_data[idx]: {class_label: (ref_image_id, ref_ann_id)}

        if len(exclude_labels) > 0:
            new_instances = []
            for ins in instances:
                if ins["bbox_label"] not in exclude_labels:
                    new_instances.append(ins)
        else:
            new_instances = instances

        ref_data_info = dict()
        for label, (ref_image_id, ref_ann_id) in ref_data.items():
            ref_img_info = self.coco.load_imgs([ref_image_id])[0]
            ref_img_info['img_id'] = ref_image_id

            ref_ann_info = self.coco.load_anns([ref_ann_id])
            parsed_data_info = self.parse_data_info({
                'raw_ann_info': ref_ann_info,
                'raw_img_info': ref_img_info
            })
            ref_data_info[label] = parsed_data_info

        return new_instances, ref_data_info

    def prepare_data(self, idx):
        """
            Each data_info is a dict with keys:
                'img_path', 'img_id', 'seg_map_path', 'height', width', 'instances'
            Instaces is a list of dict with keys:
                'ignore_flag', 'bbox', 'bbox_label', 'mask'
        """
        # idx is the sample idx for the whole image and its annotations
        data_info = self.get_data_info(idx)  # get data_info for specific idx
        image_id = data_info["img_id"]
        
        
        if self.with_ref:
            if self.ref_equal_target:
                ref_data_info = {0: data_info}
            else:
                instances = data_info["instances"] # These are the instances present in that annotated image
                new_instances, ref_data_info = self.prepare_ref_data(idx, image_id, instances)
                data_info["instances"] = new_instances

        pipelined_data = self.pipeline(data_info)

        if self.with_ref:

            if not ref_data_info.keys():
                raise ValueError("Empty data sample. Add `filter_cfg=dict(filter_empty_gt=True)`"
                                "to the dataset config to filter out empty data samples.")

            ref_data_samples = []
            ref_imgs = []
            for _, info in ref_data_info.items():
                _data = self.ref_pipeline(info)
                ref_data_samples.append(_data['data_samples'])
                ref_imgs.append(_data['inputs'][None, ...])

            ref_data = InstanceData()
            ref_data.ref_images = torch.concat(ref_imgs, dim=0)
            ref_data_instances = InstanceData.cat([x.gt_instances for x in ref_data_samples])
            ref_data.ref_data_samples = ref_data_instances
            pipelined_data['data_samples'].ref_data = ref_data

            n_ref = ref_data.ref_images.shape[0]
            # print("Data sample ", idx, "num of ref_images", n_ref)
            # if n_ref > 10:  # To avoid OOM error
            #     raise ValueError(f"TOO MANY REF IMAGES {n_ref}")
                # print(f"  ==> TOO MANY REF IMAGES {n_ref}, REPLACING WITH NEXT IDX {idx + 1}")
                # return self.prepare_data(idx + 1)

        return pipelined_data

    def load_data_list(self):
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        if self.category_ids_config is not None:
            self.cat_ids = self.category_ids_config
        else:
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        self.imgID_to_idx = {}
        for idx, img_id in enumerate(img_ids):
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            raw_img_info["info_idx"] = idx
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            self.imgID_to_idx[img_id] = idx

            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)

        self.catID_instances = {cat_id: [] for cat_id in self.cat_ids}
        for ann in self.coco.anns:
            if self.coco.anns[ann]["iscrowd"] == 1 or self.coco.anns[ann]["area"] < 32**2:
                continue
            cat_id = self.coco.anns[ann]["category_id"]
            self.catID_instances[cat_id].append(ann)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{self.ann_file}' are not unique!"

        return data_list