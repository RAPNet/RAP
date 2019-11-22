import logging
import time
import os
import json
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

import PIL.Image as Image
from pycocotools import mask as COCOmask
from ..utils.pan_utils import IdGenerator, id2rgb, save_json

from panoptic_benchmark.data.datasets.evaluation import evaluate
from panoptic_benchmark.data.datasets.evaluation.coco.ps_eval import pq_compute
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from panoptic_benchmark.modeling.roi_heads.mask_head.inference import Masker

from .combine_methods import combine_to_panoptic_for_ocfusion_acc

from panoptic_benchmark.modeling.matcher import Matcher
from panoptic_benchmark.structures.boxlist_ops import boxlist_iou
from panoptic_benchmark.structures.image_list import to_image_list


def inference_relation_ocfusion(
        cfg,
        model,
        data_loader,
        device="cuda",
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if num_devices > 1:
        print("test acc is not support multi gpu")
        exit(-1)
    dataset = data_loader.dataset

    # for relation acc
    matcher = Matcher(
        0,0,
        allow_low_quality_matches=False,
    )
    all_acc_num = 0
    all_score_acc_num = 0
    all_count = 0
    all_infer_count = 0

    # category variables
    with open("datasets/coco/panoptic_coco_categories.json", "r") as f:
        categories_list = json.load(f)
    categories = {el['id']: el for el in categories_list}
    id_generator = IdGenerator(categories)
    # sem categories
    count = 1
    sem_contiguous_ids = []
    sem_contiguous_id_to_ps_categoty_id = {}
    for l in categories_list:
        if not l["isthing"]:
            sem_contiguous_ids.append(count)
            sem_contiguous_id_to_ps_categoty_id[count] = l["id"]
            count += 1

    # compute on dataset
    model.eval()
    cpu_device = torch.device("cpu")
    masker = Masker(threshold=0.5, padding=1)
    for images, targets, image_ids in data_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            outputs = [o.to(device) for o in outputs]
        targets = [target.to(device) for target in targets]
        for image_id, output, target in zip(image_ids, outputs, targets):
            # generate pred instance id
            origin_scores = output.get_field("scores")
            keep = torch.nonzero(origin_scores > cfg.MODEL.SEMANTIC.CONFIDENCE_THR).squeeze(1)
            output = output[keep]
            try:
                match_quality_matrix = boxlist_iou(target, output)
                matched_idxs = matcher(match_quality_matrix).tolist()
            except:
                continue

            target_instance_ids = target.get_field("instance_ids").tolist()
            pred_instance_ids = []
            for idx in matched_idxs:
                if idx > 0:
                    pred_instance_ids.append(target_instance_ids[idx])
                else:
                    pred_instance_ids.append(-1)

            target_relations = target.get_field("relations")["relations"]

            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            output = output.resize((image_width, image_height))

            # detection result
            boxes = output.bbox.tolist()
            scores = output.get_field("scores").tolist()
            if output.has_field("occlusion_vals"):
                occlusion_vals = output.get_field("occlusion_vals").tolist()
            else:
                num = len(scores)
                occlusion_vals = [[0 for j in range(num)] for i in range(num)]

            labels = output.get_field("labels").tolist()
            labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
            # mask result
            masks = output.get_field("mask")
            # Masker is necessary only if masks haven't been already resized.
            if list(masks.shape[-2:]) != [image_height, image_width]:
                masks = masker(masks.expand(1, -1, -1, -1, -1), output)
                masks = masks[0]

            # construct instance results
            inst_results = []
            id_list = [i+1 for i in range(len(scores))]
            for i, box, score, instance_id, occlusion_val, label, mask in zip(id_list, boxes, scores, pred_instance_ids, occlusion_vals, labels, masks):
                inst_results.append({
                    "id": i,
                    "box": box,
                    "score": score,
                    "instance_id": instance_id,
                    "occlusion_val": occlusion_val,
                    "label": label,
                    "segmentation": COCOmask.encode(np.asfortranarray(mask[0]))
                })

            # segmentation fusion
            acc_num, score_acc_num, infer_count, count = combine_to_panoptic_for_ocfusion_acc(cfg, img_info, inst_results, id_generator, target_relations)
            all_acc_num += acc_num
            all_score_acc_num += score_acc_num
            all_count += count
            all_infer_count += infer_count
            if count > 0:
                print(all_acc_num, all_score_acc_num, all_count, all_infer_count)
    print(all_acc_num/all_count)
    print(all_score_acc_num/all_count)
    print(all_infer_count)

