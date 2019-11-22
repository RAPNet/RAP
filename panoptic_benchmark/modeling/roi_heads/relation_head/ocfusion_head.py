# Instance Relation
# author: lyujie chen
# date: 2019-06.

import torch
from torch import nn
import numpy as np
from random import sample

from panoptic_benchmark.utils.timer import Timer
from panoptic_benchmark.structures.bounding_box import BoxList
from .roi_relation_feature_extractors_ocfusion import make_roi_relation_feature_extractor
from .inference_ocfusion import make_roi_relation_post_processor


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIRelationHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_relation_feature_extractor(cfg)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.crit = nn.BCELoss()
        self.device = torch.device(self.cfg.MODEL.DEVICE)


    def forward(self, roi_mask_features, proposals, targets):
        """
        Arguments:
            roi_mask_features (list[Tensor]): feature-maps from mask head
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `relation` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        if not self.training:
            result = self.post_processor(roi_mask_features, proposals, self.feature_extractor)
            return result, {}
        
        # subsample mask occlusion pairs
        loss_relation = torch.Tensor([0.0]).to(self.device)
        ind = 0
        count = 0
        for proposal, target in zip(proposals, targets):
            occlusion_inputs = None
            occlusion_targets = None
            num = len(proposal)
            mask_feature = roi_mask_features[ind:ind+num]
            ind += num

            # get targets relations
            relation_gts = target.get_field("relations")["relations"]
            if len(relation_gts) == 0:
                continue

            # instance-wise mask and score            
            instance_id_2_mask_feature = {}
            pred_instance_ids = proposal.get_field("instance_ids")
            for i, instance_id in enumerate(pred_instance_ids):
                mask = mask_feature[i]
                score = scores[i]
                instance_id = int(instance_id)
                # mask
                if instance_id not in instance_id_2_mask_feature:
                    instance_id_2_mask_feature[instance_id] = [mask]
                else:
                    instance_id_2_mask_feature[instance_id].append(mask)

            # relation
            for relation_gt in relation_gts:
                instance_id1 = relation_gt[0]
                instance_id2 = relation_gt[1]
                if instance_id1 not in instance_id_2_mask_feature or instance_id2 not in instance_id_2_mask_feature:
                    continue
                for mask1 in instance_id_2_mask_feature[instance_id1]:
                    for mask2 in instance_id_2_mask_feature[instance_id2]:
                        # sample 1
                        occlusion_input = torch.cat((mask1, mask2), 0).unsqueeze(0)
                        if occlusion_inputs is None:
                            occlusion_inputs = occlusion_input
                        else:
                            occlusion_inputs = torch.cat((occlusion_inputs, occlusion_input), 0)
                        if occlusion_targets is None:
                            occlusion_targets = torch.tensor([1.0], device=self.device)
                        else:
                            occlusion_targets = torch.cat((occlusion_targets,torch.tensor([1.0], device=self.device)), 0)
                        # sample 2
                        occlusion_input = torch.cat((mask2, mask1), 0).unsqueeze(0)
                        occlusion_inputs = torch.cat((occlusion_inputs, occlusion_input), 0)
                        occlusion_targets = torch.cat((occlusion_targets,torch.tensor([0.0], device=self.device)), 0)

            if occlusion_targets is None:
                continue
    
            occlusion_num = occlusion_targets.shape[0]
            if occlusion_num > 128:
                sample_ind = sample(range(occlusion_num), 128)
                sample_ind = sorted(sample_ind.tolist())
                occlusion_inputs = occlusion_inputs[sample_ind]
                occlusion_targets = occlusion_targets[sample_ind]
            # feature extractor
            logits = self.feature_extractor(occlusion_inputs).view(-1)
            count += occlusion_num
            # loss
            loss_relation += self.crit(logits, occlusion_targets) * occlusion_num
                  
        # normalizaiton
        if count > 0:
            loss_relation /= count
            relation_loss_weight = self.cfg.MODEL.RELATION_LOSS_WEIGHT
            loss_relation *= relation_loss_weight
        loss_relation = torch.squeeze(loss_relation, dim=0)

        return all_proposals, dict(loss_relation=loss_relation)


def build_roi_ocfusion_head(cfg):
    return ROIRelationHead(cfg)
