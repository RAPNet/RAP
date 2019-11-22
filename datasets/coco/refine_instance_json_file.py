import os
import json
import pickle
import time
import numpy as np

with open("train_relation_result.pkl", "rb") as f:
    image_relation_result = pickle.load(f)

with open("annotations/instances_train2017.json", "r") as f:
    instance_content = json.load(f)
instance_images = instance_content["images"]
instance_anns = instance_content["annotations"]

new_annotations = []
for i, ann in enumerate(instance_anns):
    image_id = ann["image_id"]
    instance_id = ann["id"]
    if image_id in image_relation_result:
        relation_result = image_relation_result[image_id]
        if instance_id in relation_result:
            relation_list = relation_result[instance_id]
            ann["overlap"] = relation_list
        else:
            ann["overlap"] = []
    else:
        ann["overlap"] = []
    new_annotations.append(ann)

instance_content["annotations"] = new_annotations
with open("annotations/instances_train2017_with_relation.json", "w") as f:
    json.dump(instance_content, f)
