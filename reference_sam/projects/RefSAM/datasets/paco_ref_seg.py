import copy
import random

import lvis
import torch
import numpy as np

from mmengine.dataset import force_full_init
from mmengine.fileio import get_local_path
from mmengine.dataset.base_dataset import Compose
from mmengine.structures.instance_data import InstanceData

from mmdet.datasets.lvis import LVISDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class PacoLvisRefSegTrainDataset(LVISDataset):

    CLASSES = (
        'trash_can', 'handbag', 'ball', 'basket', 'belt', 'bench',
        'bicycle', 'blender', 'book', 'bottle', 'bowl', 'box', 'broom',
        'bucket', 'calculator', 'can', 'car_(automobile)', 'carton',
        'cellular_telephone', 'chair', 'clock', 'crate', 'cup', 'dog',
        'drill', 'drum_(musical_instrument)', 'earphone', 'fan',
        'glass_(drink_container)', 'guitar', 'hammer', 'hat', 'helmet',
        'jar', 'kettle', 'knife', 'ladder', 'lamp', 'laptop_computer',
        'microwave_oven', 'mirror', 'mouse_(computer_equipment)',
        'mug', 'napkin', 'newspaper', 'pan_(for_cooking)',
        'pen', 'pencil', 'pillow', 'pipe', 'plate', 'pliers',
        'remote_control', 'plastic_bag', 'scarf', 'scissors', 'screwdriver',
        'shoe', 'slipper_(footwear)', 'soap', 'sponge', 'spoon', 'stool',
        'sweater', 'table', 'tape_(sticky_cloth_or_paper)', 'telephone',
        'television_set', 'tissue_paper', 'towel', 'tray', 'vase', 'wallet',
        'watch', 'wrench', 'car_(automobile):antenna', 'chair:apron',
        'table:apron', 'chair:arm', 'bench:arm', 'chair:back', 'guitar:back',
        'remote_control:back', 'laptop_computer:back', 'bench:back',
        'telephone:back_cover', 'cellular_telephone:back_cover', 'shoe:backstay',
        'belt:bar', 'pen:barrel', 'bottle:base', 'bowl:base', 'clock:base',
        'drum_(musical_instrument):base', 'bucket:base', 'handbag:base', 'fan:base',
        'plate:base', 'television_set:base', 'basket:base', 'can:base', 'mug:base',
        'jar:base', 'soap:base', 'cup:base', 'kettle:base', 'tray:base', 'chair:base',
        'pan_(for_cooking):base', 'blender:base', 'lamp:base', 'glass_(drink_container):base',
        'laptop_computer:base_panel', 'bicycle:basket', 'telephone:bezel',
        'cellular_telephone:bezel', 'knife:blade', 'fan:blade', 'blender:blade',
        'pliers:blade', 'scissors:blade', 'plastic_bag:body', 'bottle:body', 'guitar:body',
        'bowl:body', 'drill:body', 'pencil:body', 'drum_(musical_instrument):body',
        'sweater:body', 'trash_can:body', 'scarf:body', 'bucket:body', 'handbag:body',
        'plate:body', 'calculator:body', 'can:body', 'mouse_(computer_equipment):body',
        'mug:body', 'jar:body', 'soap:body', 'towel:body', 'kettle:body',
        'glass_(drink_container):body', 'vase:body', 'dog:body', 'towel:border', 'can:bottom',
        'bucket:bottom', 'handbag:bottom', 'mug:bottom', 'bottle:bottom', 'crate:bottom',
        'bowl:bottom', 'jar:bottom', 'pan_(for_cooking):bottom', 'box:bottom', 'plate:bottom',
        'soap:bottom', 'tray:bottom', 'glass_(drink_container):bottom', 'television_set:bottom',
        'trash_can:bottom', 'carton:bottom', 'basket:bottom', 'spoon:bowl', 'fan:bracket',
        'guitar:bridge', 'broom:brush', 'broom:brush_cap', 'belt:buckle', 'watch:buckle',
        'lamp:bulb', 'car_(automobile):bumper', 'telephone:button', 'remote_control:button',
        'television_set:button', 'cellular_telephone:button', 'clock:cable', 'blender:cable',
        'lamp:cable', 'laptop_computer:cable', 'earphone:cable', 'kettle:cable',
        'laptop_computer:camera', 'fan:canopy', 'pen:cap', 'soap:cap', 'carton:cap', 'bottle:cap',
        'soap:capsule', 'bottle:capsule', 'watch:case', 'clock:case', 'pen:clip', 'soap:closure',
        'bottle:closure', 'pipe:colied_tube', 'microwave_oven:control_panel', 'bucket:cover',
        'jar:cover', 'blender:cover', 'drum_(musical_instrument):cover', 'basket:cover',
        'book:cover', 'sweater:cuff', 'blender:cup', 'clock:decoration', 'watch:dial',
        'microwave_oven:dial', 'microwave_oven:door_handle', 'bicycle:down_tube',
        'table:drawer', 'mug:drawing', 'dog:ear', 'earphone:ear_pads', 'pillow:embroidery',
        'belt:end_tip', 'pencil:eraser', 'dog:eye', 'shoe:eyelet', 'hammer:face',
        'helmet:face_shield', 'fan:fan_box', 'car_(automobile):fender', 'pencil:ferrule',
        'scissors:finger_hole', 'guitar:fingerboard', 'lamp:finial', 'clock:finial',
        'wallet:flap', 'blender:food_cup', 'dog:foot', 'vase:foot', 'ladder:foot',
        'stool:footrest', 'bicycle:fork', 'belt:frame', 'mirror:frame', 'scarf:fringes',
        'bicycle:gear', 'car_(automobile):grille', 'hammer:grip', 'pen:grip', 'watch:hand',
        'clock:hand', 'plastic_bag:handle', 'bottle:handle', 'drill:handle', 'hammer:handle',
        'scissors:handle', 'broom:handle', 'screwdriver:handle', 'bucket:handle', 'handbag:handle',
        'wrench:handle', 'pliers:handle', 'basket:handle', 'mug:handle', 'crate:handle',
        'jar:handle', 'car_(automobile):handle', 'soap:handle', 'cup:handle', 'kettle:handle',
        'knife:handle', 'spoon:handle', 'pan_(for_cooking):handle', 'blender:handle',
        'vase:handle', 'bicycle:handlebar', 'dog:head', 'hammer:head', 'wrench:head',
        'drum_(musical_instrument):head', 'bicycle:head_tube', 'earphone:headband',
        'car_(automobile):headlight', 'guitar:headstock', 'shoe:heel', 'bottle:heel',
        'plastic_bag:hem', 'sweater:hem', 'towel:hem', 'belt:hole', 'guitar:hole', 'trash_can:hole',
        'car_(automobile):hood', 'earphone:housing', 'can:inner_body', 'bucket:inner_body',
        'plastic_bag:inner_body', 'handbag:inner_body', 'mug:inner_body', 'bottle:inner_body',
        'wallet:inner_body', 'bowl:inner_body', 'jar:inner_body', 'blender:inner_body',
        'drum_(musical_instrument):inner_body', 'glass_(drink_container):inner_body',
        'trash_can:inner_body', 'cup:inner_body', 'table:inner_body', 'kettle:inner_body',
        'hat:inner_side', 'crate:inner_side', 'helmet:inner_side', 'box:inner_side',
        'microwave_oven:inner_side', 'pan_(for_cooking):inner_side', 'basket:inner_side',
        'carton:inner_side', 'tray:inner_side', 'plate:inner_wall', 'tray:inner_wall',
        'shoe:insole', 'slipper_(footwear):insole', 'pliers:jaw', 'pliers:joint', 'calculator:key',
        'guitar:key', 'laptop_computer:keyboard', 'soap:label', 'bottle:label', 'trash_can:label',
        'shoe:lace', 'pencil:lead', 'mouse_(computer_equipment):left_button', 'table:leg',
        'bench:leg', 'chair:leg', 'dog:leg', 'stool:leg', 'can:lid', 'crate:lid', 'box:lid',
        'jar:lid', 'pan_(for_cooking):lid', 'carton:lid', 'trash_can:lid', 'kettle:lid', 'fan:light',
        'shoe:lining', 'slipper_(footwear):lining', 'hat:logo', 'mouse_(computer_equipment):logo',
        'helmet:logo', 'fan:logo', 'car_(automobile):logo', 'remote_control:logo',
        'laptop_computer:logo', 'belt:loop', 'bucket:loop', 'drum_(musical_instrument):loop',
        'broom:lower_bristles', 'watch:lug', 'drum_(musical_instrument):lug', 'car_(automobile):mirror',
        'fan:motor', 'vase:mouth', 'bottle:neck', 'spoon:neck', 'vase:neck', 'soap:neck', 'dog:neck',
        'sweater:neckband', 'dog:nose', 'pipe:nozzle', 'pipe:nozzle_stem', 'tray:outer_side',
        'shoe:outsole', 'slipper_(footwear):outsole', 'book:page', 'bicycle:pedal', 'trash_can:pedal',
        'fan:pedestal_column', 'clock:pediment', 'guitar:pickguard', 'lamp:pipe', 'hat:pom_pom',
        'belt:prong', 'can:pull_tab', 'soap:punt', 'bottle:punt', 'soap:push_pull_cap', 'shoe:quarter',
        'chair:rail', 'ladder:rail', 'mouse_(computer_equipment):right_button', 'can:rim', 'bucket:rim',
        'hat:rim', 'handbag:rim', 'mug:rim', 'bowl:rim', 'helmet:rim', 'jar:rim', 'pan_(for_cooking):rim',
        'car_(automobile):rim', 'plate:rim', 'glass_(drink_container):rim', 'tray:rim',
        'drum_(musical_instrument):rim', 'trash_can:rim', 'cup:rim', 'table:rim', 'basket:rim', 'soap:ring',
        'broom:ring', 'bottle:ring', 'fan:rod', 'tissue_paper:roll', 'tape_(sticky_cloth_or_paper):roll',
        'car_(automobile):roof', 'sponge:rough_surface', 'car_(automobile):runningboard', 'bicycle:saddle',
        'telephone:screen', 'cellular_telephone:screen', 'laptop_computer:screen', 'scissors:screw',
        'mouse_(computer_equipment):scroll_wheel', 'blender:seal_ring', 'chair:seat', 'bench:seat', 'stool:seat',
        'car_(automobile):seat', 'bicycle:seat_stay', 'bicycle:seat_tube', 'lamp:shade', 'lamp:shade_cap',
        'lamp:shade_inner_side', 'broom:shaft', 'screwdriver:shank', 'table:shelf', 'sweater:shoulder',
        'soap:shoulder', 'bottle:shoulder', 'crate:side', 'guitar:side', 'box:side', 'microwave_oven:side',
        'pan_(for_cooking):side', 'television_set:side', 'carton:side', 'basket:side',
        'mouse_(computer_equipment):side_button', 'car_(automobile):sign', 'soap:sipper', 'bottle:sipper',
        'chair:skirt', 'sweater:sleeve', 'earphone:slider', 'chair:spindle', 'car_(automobile):splashboard',
        'blender:spout', 'soap:spout', 'kettle:spout', 'bottle:spout', 'car_(automobile):steeringwheel',
        'bicycle:stem', 'stool:step', 'ladder:step', 'jar:sticker', 'chair:stile', 'hat:strap', 'watch:strap',
        'helmet:strap', 'slipper_(footwear):strap', 'belt:strap', 'chair:stretcher', 'table:stretcher',
        'bench:stretcher', 'guitar:string', 'fan:string', 'blender:switch', 'lamp:switch', 'kettle:switch',
        'chair:swivel', 'bench:table_top', 'dog:tail', 'car_(automobile):taillight', 'car_(automobile):tank',
        'carton:tapering_top', 'dog:teeth', 'towel:terry_bar', 'can:text', 'plastic_bag:text', 'mug:text',
        'newspaper:text', 'jar:text', 'carton:text', 'shoe:throat', 'microwave_oven:time_display', 'spoon:tip',
        'pen:tip', 'screwdriver:tip', 'shoe:toe_box', 'slipper_(footwear):toe_box', 'shoe:tongue', 'bottle:top',
        'microwave_oven:top', 'soap:top', 'television_set:top', 'table:top', 'carton:top', 'ladder:top_cap',
        'bicycle:top_tube', 'laptop_computer:touchpad', 'car_(automobile):trunk', 'car_(automobile):turnsignal',
        'microwave_oven:turntable', 'shoe:vamp', 'slipper_(footwear):vamp', 'blender:vapour_cover',
        'hat:visor', 'helmet:visor', 'shoe:welt', 'chair:wheel', 'car_(automobile):wheel', 'trash_can:wheel',
        'table:wheel', 'bicycle:wheel', 'watch:window', 'car_(automobile):window', 'car_(automobile):windowpane',
        'car_(automobile):windshield', 'car_(automobile):wiper', 'mouse_(computer_equipment):wire',
        'sweater:yoke', 'handbag:zip'
    )

    def __init__(
        self,
        with_ref=True,
        ignore_non_exclusive=False,
        max_sampling_try=10,
        ref_pipeline=None,
        max_categories_training=999999,
        *args,
        **kwargs
    ):

        super(PacoLvisRefSegTrainDataset, self).__init__(*args, **kwargs)

        self.with_ref = with_ref
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
        exclude_labels = []
        ref_data = {}

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
        gt_labels = set([ins["bbox_label"] for ins in instances])
        ref_data, exclude_labels = self.sample_ref_data(idx, image_id, gt_labels)

        if len(exclude_labels) > 0:
            new_instances = []
            for ins in instances:
                if ins["bbox_label"] not in exclude_labels:
                    new_instances.append(ins)
        else:
            new_instances = instances

        ref_data_info = dict()
        for label, (ref_image_id, ref_ann_id) in ref_data.items():
            ref_img_info = self.lvis.load_imgs([ref_image_id])[0]
            ref_img_info['img_id'] = ref_image_id
            ref_img_info['file_name'] = ref_img_info['coco_url'].replace('http://images.cocodataset.org/', '')

            ref_ann_info = self.lvis.load_anns([ref_ann_id])
            parsed_data_info = self.parse_data_info({
                'raw_ann_info': ref_ann_info,
                'raw_img_info': ref_img_info
            })
            ref_data_info[label] = parsed_data_info

        return new_instances, ref_data_info

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        image_id = data_info["img_id"]
        if self.with_ref:
            instances = data_info["instances"]
            new_instances, ref_data_info = self.prepare_ref_data(idx, image_id, instances)
            data_info["instances"] = new_instances

        pipelined_data = self.pipeline(data_info)

        if self.with_ref:
            ref_data_samples = []
            ref_imgs = []
            for _, info in ref_data_info.items():
                _data = self.ref_pipeline(info)
                ref_data_samples.append(_data['data_samples'])
                ref_imgs.append(_data['inputs'][None, ...])

            ref_data = InstanceData()
            ref_data.ref_images = torch.concat(ref_imgs, dim=0)
            ref_data.ref_data_samples = InstanceData.cat([x.gt_instances for x in ref_data_samples])
            pipelined_data['data_samples'].ref_data = ref_data

        return pipelined_data


    def load_data_list(self):
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = lvis.LVIS(local_path)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        self.imgID_to_idx = {}
        for idx, img_id in enumerate(img_ids):
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = raw_img_info['coco_url'].replace('http://images.cocodataset.org/', '')
            raw_img_info["info_idx"] = idx
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            self.imgID_to_idx[img_id] = idx

            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)

        self.catID_instances = {cat_id: [] for cat_id in self.cat_ids}
        for ann in self.lvis.anns:
            if self.coco.anns[ann]["iscrowd"] == 1 or self.coco.anns[ann]["area"] < 16**2:
                continue
            cat_id = self.lvis.anns[ann]["category_id"]
            self.catID_instances[cat_id].append(ann)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{self.ann_file}' are not unique!"
        return data_list