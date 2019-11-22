# Instance Relation
# author: lyujie chen
# date: 2019-06.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from panoptic_benchmark.utils.timer import Timer


class RelationPostProcessor(nn.Module):
    def __init__(self):
        super(RelationPostProcessor, self).__init__()
        
    def forward(self, features, proposals, feature_extractor):
        before_num = 0
        for proposal in proposals:
            num = len(proposal)
            if num == 0:
                continue
            relation_pairs = []
            feature = features[before_num:before_num+num]
            before_num += num
            # This value (50) is determined by the GPU memory size.
            # I use 85 for NVIDIA P40 (24GB).
            if num > 50:
                for i in range(num):
                    mask1 = feature[i:i+1]
                    temp_tensor = mask1.expand(num,-1,-1,-1)
                    occlusion_inputs = torch.cat((temp_tensor, feature), dim=1)
                    if occlusion_inputs is not None:
                        logits = feature_extractor(occlusion_inputs).view(-1)
                        flag = logits.detach().cpu().numpy() > 0.5
                        relation_pairs.append(flag)

                relation_pairs = torch.from_numpy(np.array(relation_pairs, np.uint8))
            else:
                feature = feature.unsqueeze(0)
                feature = feature.expand(num,-1,-1,-1,-1)
                occlusion_inputs = torch.cat((feature.transpose(0,1), feature), dim=2)
                occlusion_inputs = occlusion_inputs.view(-1, 514, 14, 14)
                logits = feature_extractor(occlusion_inputs).view(num, num)
                relation_pairs = logits.detach() > 0.5
                relation_pairs = relation_pairs.to(torch.uint8)

            proposal.add_field("occlusion_vals", relation_pairs)
        
        return proposals

def make_roi_relation_post_processor(cfg):
    relation_post_processor = RelationPostProcessor()
    return relation_post_processor
