# Instance Relation
# author: lyujie chen
# date: 2019-06.

import torch
from torch import nn

from panoptic_benchmark.utils.timer import Timer
from panoptic_benchmark.structures.bounding_box import BoxList
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator


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
        self.predictor = make_roi_relation_predictor(cfg)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)

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

        # feature extractor
        x = self.feature_extractor(roi_mask_features)
        relation_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(relation_logits, proposals)
            return result, {}

        loss_relation = self.loss_evaluator(proposals, relation_logits, targets)
        return all_proposals, dict(loss_relation=loss_relation)


def build_roi_relation_head(cfg):
    return ROIRelationHead(cfg)
