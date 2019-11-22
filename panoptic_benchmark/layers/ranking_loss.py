# Author: jimchenhub
import torch


def ranking_loss(z_A, z_B, relation):
    """
    if relation is +1 or -1, loss is equal to torch.log(1+torch.exp(relation*(z_A-z_B)))
        when relation is +1 which means closer, we want z_A < z_B
            if z_A-z_B = -Inf, then loss = 0, if z_A > z_B, loss getting larger
        when relation is -1 which means further, we want z_A > z_B, the analysis is the same as above
        when relation is 0 which means no relation, we set loss to be common L2 loss
    """
    abs_relation = torch.abs(relation)
    return abs_relation*torch.log(1+torch.exp(relation*(z_A-z_B)))+(1-abs_relation)*(z_A-z_B)**2
