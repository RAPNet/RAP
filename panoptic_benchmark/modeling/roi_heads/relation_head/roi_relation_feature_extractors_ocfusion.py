# Instance Relation
# author: lyujie chen
# date: 2019-06.

import torch
from torch import nn
from torch.nn import functional as F

from panoptic_benchmark.layers import Conv2d


class RelationFeatureExtractor(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(self, cfg):
        super(RelationFeatureExtractor, self).__init__()

        input_channels = 514

        self.relation_oc_fcn1 = nn.Conv2d(input_channels, 512, 3, 1, 1)
        self.relation_oc_fcn2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relation_oc_fcn3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relation_oc_fcn4 = nn.Conv2d(512, 512, 3, 2, 1)
        self.relation_oc_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.relation_oc_fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for l in [self.relation_oc_fcn1, self.relation_oc_fcn2, self.relation_oc_fcn3, self.relation_oc_fcn4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.relation_oc_fc1, self.relation_oc_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = F.relu(self.relation_oc_fcn1(x))
        x = F.relu(self.relation_oc_fcn2(x))
        x = F.relu(self.relation_oc_fcn3(x))
        x = F.relu(self.relation_oc_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.relation_oc_fc1(x))
        x = self.sigmoid(self.relation_oc_fc2(x))
        return x


def make_roi_relation_feature_extractor(cfg):
    func = RelationFeatureExtractor
    return func(cfg)
