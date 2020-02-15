import copy
import json

import numpy as np
import torch
import albumentations as A

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode


class DummyAlbuMapper:
    """
    To use albumentations:
    1. Create serialized json file with augmentation config. See "sample-detection-albu-config.json"
    2. Define cfg.INPUT.ALBUMENTATIONS variable in detectron config file.

    Also I'm using detectron tools to resize and crop images 
    because they allow to use:
    ResizeShortestEdge
    RandomCrop relative to the original image size
    
    To change CROP/RESIZE parameters modify config variables:
    cfg.INPUT.CROP.[...]
    cfg.INPUT.[MAX|MIN]_SIZE_[TEST|TRAIN]
    """
    
    def __init__(self, cfg, is_train=True):
        self.aug = self._get_aug(cfg.INPUT.ALBUMENTATIONS)
        self.img_format = cfg.INPUT.FORMAT
        self.resize_gen = utils.build_transform_gen(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
        else:
            self.crop_gen = None
            
    def _get_aug(self, arg):
        with open(arg) as f:
            return A.from_dict(json.load(f))

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        img = utils.read_image(dataset_dict['file_name'], format=self.img_format)

        boxes = [ann['bbox'] for ann in dataset_dict['annotations']]
        labels = [ann['category_id'] for ann in dataset_dict['annotations']]

        augm_annotation = self.aug(image=img, bboxes=boxes, category_id=labels)

        img = augm_annotation['image']
        h, w, _ = img.shape

        augm_boxes = np.array(augm_annotation['bboxes'], dtype=np.float32)
        # sometimes bbox annotations go beyond image
        augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0, 0, 0, 0], max=[w, h, w, h])
        augm_labels = np.array(augm_annotation['category_id'])
        dataset_dict['annotations'] = [
            {
                'iscrowd': 0,
                'bbox': augm_boxes[i].tolist(),
                'category_id': augm_labels[i],
                'bbox_mode': BoxMode.XYWH_ABS,
            }
            for i in range(len(augm_boxes))
        ]
        
        if self.crop_gen:
            # image crop using detectron tools
            crop_tfm = utils.gen_crop_transform_with_instance(
                self.crop_gen.get_crop_size(img.shape[:2]),
                img.shape[:2],
                np.random.choice(dataset_dict["annotations"])
            )
            img = crop_tfm.apply_image(img)

        # image resize using detectron tools
        img, transforms = T.apply_transform_gens(self.resize_gen, img)
        if self.crop_gen:
            transforms = crop_tfm + transforms
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, img.shape[:2]
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        dataset_dict['annotations'] = annos
        instances = utils.annotations_to_instances(
            annos, img.shape[:2]
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        dataset_dict['height'] = img.shape[0]
        dataset_dict['width'] = img.shape[1]

        dataset_dict["image"] = torch.as_tensor(
            img.transpose(2, 0, 1).astype("float32")
        ).contiguous()

        return dataset_dict
