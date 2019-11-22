import torch
import torch.nn.functional as F
from torch import nn
import cv2


class SemanticFPN(nn.Module):
    """
    FCN semantic predictor based on 'panoptic FPN' paper.
    """

    def __init__(self, cfg, in_channels):
        super(SemanticFPN, self).__init__()
        self.cfg = cfg.clone()
        self.class_num = cfg.MODEL.SEMANTIC.NUM_CLASSES + 1
        group_num = 32
        
        self.semantic_seq0 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.semantic_seq1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.semantic_seq2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.semantic_seq3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.semantic_seq4 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.ReLU()
        )

        self.semantic_final = nn.Sequential(
            nn.Conv2d(128, self.class_num, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

    def forward(self, x, targets):
        sem_losses = {}

        x0 = self.semantic_seq0(x[4])
        x1 = self.semantic_seq1(x[3])
        x2 = self.semantic_seq2(x[2])
        x3 = self.semantic_seq3(x[1])
        x4 = self.semantic_seq4(x[0])
        x0 = F.interpolate(x0, size=x4.size()[-2:], mode="bilinear", align_corners=True)
        x1 = F.interpolate(x1, size=x4.size()[-2:], mode="bilinear", align_corners=True)
        x2 = F.interpolate(x2, size=x4.size()[-2:], mode="bilinear", align_corners=True)
        x3 = F.interpolate(x3, size=x4.size()[-2:], mode="bilinear", align_corners=True)

        x = x0 + x1 + x2 + x3 + x4
        x = self.semantic_final(x)
        
        if self.training:
            # calculate loss
            loss_semantic = 0.0
            batch_count = 0
            for i in range(len(targets)):
                label = targets[i].get_field("semantic_label").copy()
                x_i = F.interpolate(x[i:i+1], size=label.shape, mode='bilinear', align_corners=True)
                label = torch.LongTensor(label).unsqueeze(0)
                label = label.to(device=self.cfg.MODEL.DEVICE)
                count = torch.sum(label > 0)
                loss_semantic += F.cross_entropy(x_i, label, ignore_index=0, reduction="sum")
                batch_count += count
            if batch_count > 0:
                loss_semantic /= batch_count
            loss_semantic *= self.cfg.MODEL.SEMANTIC.LOSS_WEIGHT
            sem_losses.update(dict(loss_semantic=loss_semantic))
            return None, sem_losses
        else:
            return x, {}

def build_semantic_head(cfg, in_channels):
    semantic_head = SemanticFPN(cfg, in_channels)
    return semantic_head
