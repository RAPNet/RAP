# Instance Relation
# author: lyujie chen
# date: 2019-06.

import torch
from torch.nn import functional as F
from random import sample

from panoptic_benchmark.layers.ranking_loss import ranking_loss


class RelationLossComputation(object):
    def __init__(self, device, loss_weight):
        self.device = device
        self.loss_weight = loss_weight
    
    def __call__(self, proposals, pred_vals, targets):
        pos = 0
        losses = torch.Tensor([0.0]).to(self.device)
        count = 0
        for proposal, target in zip(proposals, targets):
            pred_val = pred_vals[pos:pos+len(proposal)]
            pos += len(proposal)
            # get targets relations
            relation_gt = target.get_field("relations")["relations"]
            for k,v in relation_gt.items():
                relation_gt[k] = v.to(self.device)
            # get proposal prediction
            pred_instance_ids = proposal.get_field("instance_ids")
            for i in range(len(pred_instance_ids)):
                ins_i = int(pred_instance_ids[i])
                val_i = pred_val[i]
                for j in range(i, len(pred_instance_ids)):
                    ins_j = int(pred_instance_ids[j])
                    val_j = pred_val[j]
                    if (ins_i, ins_j) in relation_gt:
                        losses += ranking_loss(val_i, val_j, relation_gt[(ins_i, ins_j)])
                        count += 1
                    elif (ins_j, ins_i) in relation_gt:
                        losses += ranking_loss(val_j, val_i, relation_gt[(ins_j, ins_i)])
                        count += 1
        # normalizaiton
        if count > 0:
            losses /= count
            losses *= self.loss_weight
        losses = torch.squeeze(losses, dim=0)
        return losses


def make_roi_relation_loss_evaluator(cfg):
    loss_weight = cfg.MODEL.RELATION_LOSS_WEIGHT
    device = torch.device(cfg.MODEL.DEVICE)
    loss_evaluator = RelationLossComputation(device, loss_weight)

    return loss_evaluator


