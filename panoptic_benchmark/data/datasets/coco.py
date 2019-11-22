import torch
import torchvision

from panoptic_benchmark.structures.bounding_box import BoxList
from panoptic_benchmark.structures.segmentation_mask import SegmentationMask
from panoptic_benchmark.structures.keypoint import PersonKeypoints
from panoptic_benchmark.structures.overlap_relation import OverlapRelation
from panoptic_benchmark.structures.instance_id import InstanceId

import cv2
import json


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, cfg, ann_file, is_train, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        self.cfg = cfg
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.transforms = transforms
        self.is_train = is_train


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # if no target
        if len(anno) == 0:
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, idx

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        # ----------------------------------
        # add semantic annotations
        if "coco" in self.cfg.DATASETS.TRAIN[0]:
            file_name = self.get_img_info(idx)["file_name"].replace(".jpg", ".png")
            if self.is_train:
                semantic_base_folder = "datasets/coco/annotations/semantic_train2017/"
            else:
                semantic_base_folder = "datasets/coco/annotations/semantic_val2017/"
            semantic_file_name = semantic_base_folder + file_name
            # read semantic image
            semantic_image = cv2.imread(semantic_file_name)
            semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2GRAY)
            target.add_field("semantic_label", semantic_image)
        # ----------------------------------

        # ----------------------------------
        # add instance id field
        ins_ids = [obj["id"] for obj in anno]
        instance_ids = InstanceId(ins_ids)
        target.add_field("instance_ids", instance_ids)
        if self.cfg.MODEL.RELATION_ON:
            # add relation field
            overlaps = [obj["overlap"] for obj in anno]
            relations = OverlapRelation(overlaps, instance_ids)
            target.add_field("relations", relations)
        # ----------------------------------

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
