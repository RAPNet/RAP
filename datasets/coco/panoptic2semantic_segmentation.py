import json
import os
import cv2
import pickle
import numpy as np
import time
import datetime
import threading


panoptic_file = "annotations/panoptic_val2017.json"
panoptic_image_folder = "annotations/panoptic_val2017/"
output_folder = "annotations/semantic_val2017/"


def translate(thread_name, image_list, panoptic_image_folder, output_folder, 
              image_anns, category_id_to_contiguous_id):
    for i, name in enumerate(image_list):
        panop_image = os.path.join(panoptic_image_folder, name)
        anns = image_anns[name]
        # calculate each pixel id
        image = cv2.imread(panop_image)
        output_image = np.zeros(image.shape[:2])
        id_image = image[:,:,2] + 256*image[:,:,1] + 256*256*image[:,:,0]
        for ann in anns:
            output_image[id_image == ann["id"]] = category_id_to_contiguous_id[ann["category_id"]]
        output_file_name = os.path.join(output_folder, name)
        cv2.imwrite(output_file_name, output_image)
        if i % 100 == 0:
            time_stamp = datetime.datetime.now()
            print(time_stamp.strftime('%Y.%m.%d-%H:%M:%S'), thread_name, i, output_file_name)


if __name__ == "__main__":
    with open(panoptic_file, "r") as f:
        panoptic_content = json.load(f)
    image_list = os.listdir(panoptic_image_folder)

    # process annotations
    ann_list = panoptic_content["annotations"]
    image_anns = {a["file_name"]: a["segments_info"] for a in ann_list}

    # read categories and corresponding color
    with open("panoptic_coco_categories.json", "r") as f:
        categories = json.load(f)
    category_ids = [l["id"] for l in categories]
    category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(category_ids)}
    category_id_to_contiguous_id[0] = 0

    threads = []

    thread_num = 20
    each_num = 250
    for i in range(thread_num):
        t = threading.Thread(target=translate, args=(f"thread-{i+1}", image_list[each_num*i:(i+1)*each_num], panoptic_image_folder, output_folder, image_anns, category_id_to_contiguous_id))
        threads.append(t)
    
    for t in threads:
        t.start()
