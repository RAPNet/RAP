# Instance Relation
# author: lyujie chen
# date: 2019-06.

from torch import nn
from torch.nn import functional as F

from panoptic_benchmark.layers import Conv2d
from panoptic_benchmark.layers import ConvTranspose2d


class RelationPredictor(nn.Module):
    def __init__(self, cfg):
        super(RelationPredictor, self).__init__()

        self.relation_val = nn.Linear(1024, 1)

        nn.init.normal_(self.relation_val.weight, mean=0, std=0.01)
        nn.init.constant_(self.relation_val.bias, 0)


    def forward(self, x):
        relation_val = self.relation_val(x)
        return relation_val


def make_roi_relation_predictor(cfg):
    return RelationPredictor(cfg)
