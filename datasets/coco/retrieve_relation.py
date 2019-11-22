import os
import json
import time
import numpy as np
import cv2
import pickle
import logging


panoptic_image_folder = "annotations/panoptic_train2017/"
image_folder = "images/train2017/"


# read instance json
with open("annotations/instances_train2017.json", "r") as f:
    instance_content = json.load(f)
instance_images = instance_content["images"]
instance_anns = instance_content["annotations"]

instance_image_file_name = {image["id"]: image["file_name"] for image in instance_images}

# read panoptic json
with open("annotations/panoptic_train2017.json", "r") as f:
    panoptic_content = json.load(f)
panoptic_images = panoptic_content["images"]
panoptic_anns = panoptic_content["annotations"]
category_thing = [a["id"] for a in panoptic_content["categories"] if a["isthing"]]

# read image and anns
image2instanceann = {}
for ann in instance_anns:
    image_id = ann["image_id"]
    if image_id not in image2instanceann:
        image2instanceann[image_id] = [ann]
    else:
        image2instanceann[image_id].append(ann)

image2panopticann = {}
for ann in panoptic_anns:
    image_id = ann["image_id"]
    image2panopticann[image_id] = ann


# generate relation 
count = 0
t1 = time.time()
covered_image_count = 0
relation_count = 0
image_relation_result = {}
for image in panoptic_images:
    image_id = image["id"]
    count += 1
    if count % 100 == 0:
        t2 = time.time()
        print(f"{t2-t1}, {count}, {covered_image_count}, {relation_count}")
        t1 = time.time()
    # if ann not exist
    if image_id not in image2instanceann or image_id not in image2panopticann:
        continue
    # read panoptic image
    panoptic_filename = os.path.join(panoptic_image_folder, image2panopticann[image_id]["file_name"])
    image_filename = os.path.join(image_folder, image["file_name"])

    rgb = cv2.imread(image_filename)
    img = cv2.imread(panoptic_filename)
    width = image["width"]
    height = image["height"]

    # instance and panoptic annotations
    instance_ann = image2instanceann[image_id]
    panoptic_ann = [a for a in image2panopticann[image_id]["segments_info"] if a["category_id"] in category_thing ]

    panoptic_id_2_instance_id = {}
    instance_id_2_panoptic_id = {}
    for ann in panoptic_ann:
        bbox = np.array(ann["bbox"])
        instance_id = None
        min_dist = 9999
        for ins_ann in instance_ann:
            ins_bbox = np.array(ins_ann["bbox"])
            dist = np.sqrt(np.sum((bbox-ins_bbox)**2))
            if dist < min_dist:
                min_dist = dist
                instance_id = ins_ann["id"]
        # if conflict
        if instance_id in instance_id_2_panoptic_id:
            del panoptic_id_2_instance_id[instance_id_2_panoptic_id[instance_id]]
            del instance_id_2_panoptic_id[instance_id]
        else:   # not conflict
            instance_id_2_panoptic_id[instance_id] = ann["id"]
            panoptic_id_2_instance_id[ann["id"]] = instance_id

    # delete no use ann
    if len(instance_id_2_panoptic_id) < len(panoptic_ann):
        new_instance_ann = []
        new_panoptic_ann = []
        for instance in panoptic_ann:
            if instance["id"] in panoptic_id_2_instance_id: 
                new_panoptic_ann.append(instance)
        for instance in instance_ann:
            if instance["id"] in instance_id_2_panoptic_id: 
                new_instance_ann.append(instance)
        panoptic_ann = new_panoptic_ann
        instance_ann = new_instance_ann

    # more instance in instance ann
    if len(instance_ann) > len(panoptic_ann):
        new_instance_ann = []
        for instance in instance_ann:
            if instance["id"] in instance_id_2_panoptic_id:
                new_instance_ann.append(instance)
        instance_ann = new_instance_ann

    assert len(panoptic_ann) == len(instance_ann)
    assert len(np.unique([a for a, b in panoptic_id_2_instance_id.items()])) == len([b for a, b in panoptic_id_2_instance_id.items()])

    # generate panoptic instance masks
    pano_crowd = {ann["id"]: ann["iscrowd"] for ann in panoptic_ann}
    inst_crowd = {ann["id"]: ann["iscrowd"] for ann in instance_ann}

    panoptic_masks = {}
    for ann in panoptic_ann:
        if ann["iscrowd"] == 0 and inst_crowd[panoptic_id_2_instance_id[ann["id"]]] == 0:
            ann_img = np.zeros((height, width)).astype(np.uint8)
            bbox = ann["bbox"]
            instance_id = ann["id"]
            # panoptic mask area
            area = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]].astype(np.int32)
            area_sum = area[:,:,0]*256*256+area[:,:,1]*256+area[:,:,2]
            mask = (area_sum == instance_id).astype(np.uint8)
            ann_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = mask
            panoptic_masks[instance_id] = ann_img

    # generate instance masks
    instance_masks = {}
    for ann in instance_ann:
        if ann["iscrowd"] == 0 and pano_crowd[instance_id_2_panoptic_id[ann["id"]]] == 0:
            instance_id = ann["id"]
            ann_img = np.zeros((height, width)).astype(np.uint8)
            all_seg = ann["segmentation"]
            all_seg = [[[seg[i*2], seg[i*2+1]] for i in range(len(seg)//2)] for seg in all_seg]
            for seg in all_seg:
                seg = np.array([seg]).astype(np.int)
                cv2.fillPoly(ann_img, seg, 1)
            instance_masks[instance_id] = ann_img
    
    assert len(panoptic_masks) == len(instance_masks)

    # compare instances
    instance_id_list = list(instance_masks.keys())
    mask_areas = [np.sum(instance_masks[i] > 0) for i in instance_id_list]
    draw_img = np.zeros((height, width, 3)).astype(np.uint8)
    relation_result = {}
    each_relation_count = 1
    for i in range(len(instance_id_list)):
        id1 = instance_id_list[i]
        mask1 = instance_masks[id1]
        area1 = mask_areas[i]
        for j in range(i+1, len(instance_id_list)):
            id2 = instance_id_list[j]
            mask2 = instance_masks[id2]
            area2 = mask_areas[j]
            overlap = cv2.bitwise_and(mask1, mask2)
            area = np.sum(overlap)
            # check whether manually error
            if area/area1 > 0.2 or area/area2 > 0.2:
                #overlay = rgb.copy()
                #output = rgb.copy()
                # check relation based on panoptic masks
                pano_mask1 = panoptic_masks[instance_id_2_panoptic_id[id1]]
                pano_mask2 = panoptic_masks[instance_id_2_panoptic_id[id2]]
                pano_area1 = np.sum(pano_mask1 > 0)
                pano_area2 = np.sum(pano_mask2 > 0)
                delta_area1 = area1 - pano_area1
                delta_area2 = area2 - pano_area2
                # print(delta_area1, delta_area2)
                if delta_area1 < delta_area2:
                    # print(id1, "is closer than", id2)
                    if id2 in relation_result:
                        relation_result[id2].append(id1)
                    else:
                        relation_result[id2] = [id1]
                    #overlay[instance_masks[id2]>0] = (200,0,0)
                    #overlay[instance_masks[id1]>0] = (0,0,200)
                elif delta_area2 < delta_area1:
                    # print(id2, "is closer than", id1)
                    if id1 in relation_result:
                        relation_result[id1].append(id2)
                    else:
                        relation_result[id1] = [id2]
                    #overlay[instance_masks[id1]>0] = (200,0,0)
                    #overlay[instance_masks[id2]>0] = (0,0,200)
                #cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
                #cv2.imwrite("train_relation_vis/"+image["file_name"].split(".")[0]+"_"+str(each_relation_count)+".png", output)
                each_relation_count += 1
        
    each_count = sum([len(b) for a, b in relation_result.items()])
    if each_count > 0:
        covered_image_count += 1
    relation_count += each_count

    image_relation_result[image_id] = relation_result
    
print(covered_image_count)
print(relation_count)
with open("train_relation_result.pkl", "wb") as f:
    pickle.dump(image_relation_result, f)

# val
# 0.2 and 0.2
# 2025
# 6279
 
# train
# 0.2 and 0.2
# 48719
# 146711

