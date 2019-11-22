import os
import PIL.Image as Image
from pycocotools import mask as COCOmask
from ..utils.pan_utils import IdGenerator, id2rgb, save_json
import numpy as np
import cv2
import copy
import torch


def combine_to_panoptic_heuristic(cfg, img_info, inst_results, sem_results, id_generator, segmentations_folder):
    overlap_thr = cfg.MODEL.SEMANTIC.OVERLAP_THR
    confidence_thr = cfg.MODEL.SEMANTIC.CONFIDENCE_THR
    stuff_area_limit = cfg.MODEL.SEMANTIC.STUFF_AREA_LIMIT

    # remove instance with score less than confidence_thr
    inst_results = [inst for inst in inst_results if not inst["score"] < confidence_thr]

    # sort instance by score
    inst_results = sorted(inst_results, key=lambda el: -el['score'])

    # construct panoptic segmentation image
    pan_segm_id = np.zeros((img_info['height'], img_info['width']), dtype=np.uint32)
    used = None
    annotation = {}

    try:
        annotation['image_id'] = int(img_info["id"])
    except Exception:
        annotation['image_id'] = img_info["id"]

    annotation['file_name'] = img_info["file_name"].replace('.jpg', '.png')

    segments_info = []
    # --- combine instance one by one
    for ann in inst_results:
        area = COCOmask.area(ann['segmentation'])
        if area == 0:
            continue
        if used is None:
            intersect = 0
            used = copy.deepcopy(ann['segmentation'])
        else:
            intersect = COCOmask.area(
                COCOmask.merge([used, ann['segmentation']], intersect=True)
            )
        if intersect / area > overlap_thr:
            continue
        used = COCOmask.merge([used, ann['segmentation']], intersect=False)

        mask = COCOmask.decode(ann['segmentation']) == 1
        if intersect != 0:
            mask = np.logical_and(pan_segm_id == 0, mask)
        #print(ann["id"])
        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask] = segment_id
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = ann['label']
        segments_info.append(panoptic_ann)

    # --- combine semantic area one by one
    for ann in sem_results:
        mask = COCOmask.decode(ann['segmentation']) == 1
        mask_left = np.logical_and(pan_segm_id == 0, mask)
        if mask_left.sum() < stuff_area_limit:
            continue
        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask_left] = segment_id
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = ann['label']
        segments_info.append(panoptic_ann)

    annotation['segments_info'] = segments_info
    Image.fromarray(id2rgb(pan_segm_id)).save(os.path.join(segmentations_folder, annotation['file_name']))

    return annotation


def combine_to_panoptic_RAP(cfg, img_info, inst_results, sem_results, id_generator, segmentations_folder):
    overlap_thr = cfg.MODEL.SEMANTIC.OVERLAP_THR
    confidence_thr = cfg.MODEL.SEMANTIC.CONFIDENCE_THR
    stuff_area_limit = cfg.MODEL.SEMANTIC.STUFF_AREA_LIMIT
    large_overlap_thr = cfg.MODEL.SEMANTIC.LARGE_OVERLAP_THR

    # remove instance with score less than confidence_thr
    inst_results = [inst for inst in inst_results if not inst["score"] < confidence_thr]

    # sort instance by score
    inst_results = sorted(inst_results, key=lambda el: -el['score'])

    # construct panoptic segmentation image
    pan_segm_id = np.zeros((img_info['height'], img_info['width']), dtype=np.uint32)
    used_label = np.zeros((img_info['height'], img_info['width']), np.int32)
    used_relation_vals = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
    used_relation_vals[:,:] = float("Inf")
    annotation = {}

    try:
        annotation['image_id'] = int(img_info["id"])
    except Exception:
        annotation['image_id'] = img_info["id"]

    annotation['file_name'] = img_info["file_name"].replace('.jpg', '.png')

    segments_info = []
    # --- combine instance one by one
    #inst_count = 0
    for ann in inst_results:
        area = COCOmask.area(ann['segmentation'])
        mask = COCOmask.decode(ann['segmentation']) == 1

        if area == 0:
            continue

        intersect_mask = np.logical_and(pan_segm_id > 0, mask)
        same_label_intersect_mask = np.logical_and(intersect_mask, used_label==ann["label"])
        intersect = np.sum(intersect_mask)
        same_label_intersect = np.sum(same_label_intersect_mask)

        if same_label_intersect / area > overlap_thr:
            continue

        is_valid = True

        if intersect != 0:
            intersect_ids = np.unique(pan_segm_id[intersect_mask])
            for intersect_id in intersect_ids:
                id_area = np.sum(pan_segm_id==intersect_id)
                intersect_id_mask = np.logical_and((pan_segm_id == intersect_id), intersect_mask)
                intersect_id_area= np.sum(intersect_id_mask)

                if intersect_id_area/area >= large_overlap_thr and intersect_id_area/id_area >= large_overlap_thr:
                   is_valid = False
                   break

            mask = np.logical_and(used_relation_vals > ann["relation_val"], mask)

        if not is_valid:
           continue
        if np.sum(mask) / area <= overlap_thr:
            continue

        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask] = segment_id
        used_label[mask] = ann["label"]
        used_relation_vals[mask] = ann["relation_val"]
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = ann['label']
        segments_info.append(panoptic_ann)

    # check if instance still exist
    refine_segments_info = []
    for panoptic_ann in segments_info:
        segment_id = panoptic_ann['id']
        area = np.sum(pan_segm_id == segment_id)
        if area > 0:
            refine_segments_info.append(panoptic_ann)
    segments_info = refine_segments_info

    # --- combine semantic area one by one
    for ann in sem_results:
        mask = COCOmask.decode(ann['segmentation']) == 1
        mask_left = np.logical_and(pan_segm_id == 0, mask)
        if mask_left.sum() < stuff_area_limit:
            continue
        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask_left] = segment_id
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = ann['label']
        segments_info.append(panoptic_ann)

    annotation['segments_info'] = segments_info
    Image.fromarray(id2rgb(pan_segm_id)).save(os.path.join(segmentations_folder, annotation['file_name']))

    return annotation


def combine_to_panoptic_ocfusion(cfg, img_info, inst_results, sem_results, id_generator, segmentations_folder):
    overlap_thr = cfg.MODEL.SEMANTIC.OVERLAP_THR
    confidence_thr = cfg.MODEL.SEMANTIC.CONFIDENCE_THR
    stuff_area_limit = cfg.MODEL.SEMANTIC.STUFF_AREA_LIMIT

    # remove instance with score less than confidence_thr
    inst_results = [inst for inst in inst_results if not inst["score"] < confidence_thr]

    # sort instance by score
    inst_results = sorted(inst_results, key=lambda el: -el['score'])

    # construct panoptic segmentation image
    pan_segm_id = np.zeros((img_info['height'], img_info['width']), dtype=np.uint32)
    used = np.zeros((img_info['height'], img_info['width']), np.int32)
    annotation = {}

    try:
        annotation['image_id'] = int(img_info["id"])
    except Exception:
        annotation['image_id'] = img_info["id"]

    annotation['file_name'] = img_info["file_name"].replace('.jpg', '.png')

    segments_info = []
    # --- combine instance one by one
    for ann in inst_results:
        area = COCOmask.area(ann['segmentation'])
        mask = COCOmask.decode(ann['segmentation']) == 1
        inst_id = ann["id"]
        occlusion_val = ann["occlusion_val"]

        if area == 0:
            continue

        intersect_mask = np.logical_and((used > 0), mask)
        intersect = np.sum(intersect_mask)

        if intersect != 0:
            mask = np.logical_and(used==0, mask)
            intersect_ids = np.unique(used[intersect_mask])
            for intersect_id in intersect_ids:
                id_area = np.sum(used==intersect_id)
                intersect_id_mask = np.logical_and((used == intersect_id), intersect_mask)
                intersect_id_area= np.sum(intersect_id_mask)
                if intersect_id_area/area >= 0.2 or intersect_id_area/id_area >= 0.2:
                    val = occlusion_val[intersect_id-1]
                    if val:
                        mask[intersect_id_mask] = True
            if np.sum(mask)/area <= overlap_thr:
                continue

        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask] = segment_id
        used[mask] = inst_id
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = ann['label']
        segments_info.append(panoptic_ann)

    # check if instance still exist
    refine_segments_info = []
    for panoptic_ann in segments_info:
        segment_id = panoptic_ann['id']
        area = np.sum(pan_segm_id == segment_id)
        if area > 0:
            refine_segments_info.append(panoptic_ann)
    segments_info = refine_segments_info

    # --- combine semantic area one by one
    for ann in sem_results:
        mask = COCOmask.decode(ann['segmentation']) == 1
        mask_left = np.logical_and(pan_segm_id == 0, mask)
        if mask_left.sum() < stuff_area_limit:
            continue
        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask_left] = segment_id
        panoptic_ann = {}
        panoptic_ann['id'] = segment_id
        panoptic_ann['category_id'] = ann['label']
        segments_info.append(panoptic_ann)

    annotation['segments_info'] = segments_info
    Image.fromarray(id2rgb(pan_segm_id)).save(os.path.join(segmentations_folder, annotation['file_name']))

    return annotation


def combine_to_panoptic_for_acc(cfg, img_info, inst_results, id_generator, target_relations):
    overlap_thr = cfg.MODEL.SEMANTIC.OVERLAP_THR
    confidence_thr = cfg.MODEL.SEMANTIC.CONFIDENCE_THR
    stuff_area_limit = cfg.MODEL.SEMANTIC.STUFF_AREA_LIMIT
    large_overlap_thr = cfg.MODEL.SEMANTIC.LARGE_OVERLAP_THR

    # remove instance with score less than confidence_thr
    inst_results = [inst for inst in inst_results if not inst["score"] < confidence_thr]
    # sort instance by score
    inst_results = sorted(inst_results, key=lambda el: -el['score'])
    # construct panoptic segmentation image
    pan_segm_id = np.zeros((img_info['height'], img_info['width']), dtype=np.uint32)
    used_label = np.zeros((img_info['height'], img_info['width']), np.int32)
    used_relation_vals = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
    used_relation_vals[:,:] = float("Inf")

    segment_id_to_instance_id = {}
    segment_id_to_pred_val = {}
    count = 0
    infer_count = 0
    acc_num = 0
    score_acc_num = 0

    # --- combine instance one by one
    for ann in inst_results:
        area = COCOmask.area(ann['segmentation'])
        mask = COCOmask.decode(ann['segmentation']) == 1

        if area == 0:
            continue

        intersect_mask = np.logical_and(pan_segm_id > 0, mask)
        same_label_intersect_mask = np.logical_and(intersect_mask, used_label==ann["label"])
        intersect = np.sum(intersect_mask)
        same_label_intersect = np.sum(same_label_intersect_mask)

        if same_label_intersect / area > overlap_thr:
            continue

        is_valid = True
        if intersect != 0:
            intersect_ids = np.unique(pan_segm_id[intersect_mask])
            for intersect_id in intersect_ids:
                id_area = np.sum(pan_segm_id==intersect_id)
                intersect_id_mask = np.logical_and((pan_segm_id == intersect_id), intersect_mask)
                intersect_id_area= np.sum(intersect_id_mask)

                if intersect_id_area/area >= large_overlap_thr and intersect_id_area/id_area >= large_overlap_thr:
                   is_valid = False
                   break
                
                # calculate acc
                instance_id1 = segment_id_to_instance_id[intersect_id]
                instance_id2 = ann["instance_id"]
                pred_val1 = segment_id_to_pred_val[intersect_id]
                pred_val2 = ann["relation_val"]
                if (instance_id1, instance_id2) in target_relations:
                    if pred_val1 < pred_val2:
                        acc_num += 1
                    score_acc_num += 1
                    count += 1
                elif (instance_id2, instance_id1) in target_relations:
                    if pred_val2 < pred_val1:
                        acc_num += 1
                    count += 1

            mask = np.logical_and(used_relation_vals > ann["relation_val"], mask)
            if is_valid:
                infer_count += 1

        if np.sum(mask) / area <= overlap_thr:
            continue

        if not is_valid:
           continue

        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask] = segment_id
        used_label[mask] = ann["label"]
        used_relation_vals[mask] = ann["relation_val"]
        segment_id_to_instance_id[segment_id] = ann["instance_id"]
        segment_id_to_pred_val[segment_id] = ann["relation_val"]

    return acc_num, score_acc_num, infer_count, count


def combine_to_panoptic_for_ocfusion_acc(cfg, img_info, inst_results, id_generator, target_relations):
    overlap_thr = cfg.MODEL.SEMANTIC.OVERLAP_THR
    confidence_thr = cfg.MODEL.SEMANTIC.CONFIDENCE_THR
    stuff_area_limit = cfg.MODEL.SEMANTIC.STUFF_AREA_LIMIT

    # remove instance with score less than confidence_thr
    inst_results = [inst for inst in inst_results if not inst["score"] < confidence_thr]
    # sort instance by score
    inst_results = sorted(inst_results, key=lambda el: -el['score'])
    # construct panoptic segmentation image
    pan_segm_id = np.zeros((img_info['height'], img_info['width']), dtype=np.uint32)
    used = np.zeros((img_info['height'], img_info['width']), np.int32)

    id_to_instance_id = {}
    count = 0
    infer_count = 0
    acc_num = 0
    score_acc_num = 0
    # --- combine instance one by one
    for ann in inst_results:
        area = COCOmask.area(ann['segmentation'])
        mask = COCOmask.decode(ann['segmentation']) == 1
        inst_id = ann["id"]
        occlusion_val = ann["occlusion_val"]

        if area == 0:
            continue

        intersect_mask = np.logical_and((used > 0), mask)
        intersect = np.sum(intersect_mask)

        if intersect != 0:
            mask = np.logical_and(used==0, mask)
            intersect_ids = np.unique(used[intersect_mask])
            for intersect_id in intersect_ids:
                id_area = np.sum(used==intersect_id)
                intersect_id_mask = np.logical_and((used == intersect_id), intersect_mask)
                intersect_id_area= np.sum(intersect_id_mask)
                
                val = occlusion_val[intersect_id-1]
                # calculate  acc
                instance_id1 = id_to_instance_id[intersect_id]
                instance_id2 = ann["instance_id"]
                if (instance_id1, instance_id2) in target_relations:
                    if not val:
                        acc_num += 1
                    score_acc_num += 1
                    count += 1
                elif (instance_id2, instance_id1) in target_relations:
                    if val:
                        acc_num += 1
                    count += 1

                if intersect_id_area/area >= 0.2 or intersect_id_area/id_area >= 0.2:
                    infer_count += 1
                    if val:
                        mask[intersect_id_mask] = True

            if np.sum(mask)/area <= overlap_thr:
                continue

        segment_id = id_generator.get_id(ann['label'])
        pan_segm_id[mask] = segment_id
        used[mask] = inst_id
        id_to_instance_id[inst_id] = ann["instance_id"]

    return acc_num, score_acc_num, infer_count, count

