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
from .combine_methods import combine_to_panoptic_heuristic, combine_to_panoptic_ocfusion


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = []
    for p in all_predictions:
        predictions.extend(p)
    return predictions


def inference_ps_ocfusion(
        cfg,
        model,
        data_loader,
        dataset_name,
        device="cuda",
        output_folder=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("panoptic_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()

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

    if cfg.MODEL.SEMANTIC.COMBINE_METHOD == "heuristic":
        save_dir_symbol = "_heuristic"
    elif cfg.MODEL.SEMANTIC.COMBINE_METHOD == "ocfusion":
        save_dir_symbol = "_ocfusion"

    total_timer.tic()
    # compute on dataset
    model.eval()
    panoptic_json = []
    segmentations_folder = os.path.join(output_folder, f"pred_result{save_dir_symbol}")
    if os.path.exists(segmentations_folder) is False:
        os.makedirs(segmentations_folder)
    panoptic_json_file = os.path.join(output_folder, f"panoptic_pred_val2017{save_dir_symbol}.json")
    cpu_device = torch.device("cpu")
    masker = Masker(threshold=0.5, padding=1)
    for images, targets, image_ids in tqdm(data_loader):
        images = images.to(device)
        if inference_timer:
            inference_timer.tic()
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        # process image one by one
        for image_id, prediction in zip(image_ids, output):
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width, image_height))

            # detection result
            boxes = prediction.bbox.tolist()
            scores = prediction.get_field("scores").tolist()

            if prediction.has_field("occlusion_vals"):
                occlusion_vals = prediction.get_field("occlusion_vals").tolist()
            else:
                num = len(scores)
                occlusion_vals = [[0 for j in range(num)] for i in range(num)]

            labels = prediction.get_field("labels").tolist()
            labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
            # mask result
            masks = prediction.get_field("mask")
            # Masker is necessary only if masks haven't been already resized.
            if list(masks.shape[-2:]) != [image_height, image_width]:
                masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
                masks = masks[0]

            # semantic result
            semantic_result = prediction.get_field("semantic_result")
            semantic_result = F.interpolate(semantic_result, size=(image_height, image_width), mode='bilinear', align_corners=True)
            semantic_result = F.softmax(semantic_result, dim=1)
            semantic_result = torch.argmax(semantic_result, dim=1)
            #print(torch.sum(semantic_result))

            # construct instance results
            inst_results = []
            id_list = [i+1 for i in range(len(scores))]
            for i, box, score, occlusion_val, label, mask in zip(id_list, boxes, scores, occlusion_vals, labels, masks):
                inst_results.append({
                    "id": i,
                    "box": box,
                    "score": score,
                    "occlusion_val": occlusion_val,
                    "label": label,
                    "segmentation": COCOmask.encode(np.asfortranarray(mask[0]))
                })

            # construct semantic results
            sem_results = []
            for sem_id in sem_contiguous_ids:
                segmentation = (semantic_result == sem_id).numpy()[0]
                if np.sum(segmentation) > 0:
                    sem_results.append({
                        "segmentation": COCOmask.encode(np.asfortranarray(segmentation)),
                        "label": sem_contiguous_id_to_ps_categoty_id[sem_id]
                    })

            # segmentation fusion
            if cfg.MODEL.SEMANTIC.COMBINE_METHOD == "heuristic":
                annotation = combine_to_panoptic_heuristic(cfg, img_info, inst_results, sem_results,
                                                                id_generator, segmentations_folder)
            elif cfg.MODEL.SEMANTIC.COMBINE_METHOD == "ocfusion":
                annotation = combine_to_panoptic_ocfusion(cfg, img_info, inst_results, sem_results,
                                                                id_generator, segmentations_folder)
            panoptic_json.append(annotation)

            if inference_timer:
                torch.cuda.synchronize()
                inference_timer.toc()

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    panoptic_json = _accumulate_predictions_from_multiple_gpus(panoptic_json)
    if not is_main_process():
        return

    if output_folder:
        coco_d = {}
        coco_d['annotations'] = panoptic_json
        coco_d['categories'] = list(categories.values())
        save_json(coco_d, panoptic_json_file)

    gt_folder = "datasets/coco/annotations/panoptic_val2017"
    gt_json_file = "datasets/coco/annotations/panoptic_val2017.json"

    return pq_compute(gt_json_file=gt_json_file,
                      pred_json_file=panoptic_json_file,
                      gt_folder=gt_folder,
                      pred_folder=segmentations_folder,
                      maskiou_on=maskiou_on)
