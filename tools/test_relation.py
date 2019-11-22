import argparse
import os

import torch
from panoptic_benchmark.config import cfg
from panoptic_benchmark.data import make_data_loader
from panoptic_benchmark.engine.inference_ps_relation import inference_relation
from panoptic_benchmark.engine.inference_ps_relation_ocfusion import inference_relation_ocfusion
from panoptic_benchmark.modeling.detector import build_detection_model
from panoptic_benchmark.utils.checkpoint import DetectronCheckpointer
from panoptic_benchmark.utils.comm import synchronize, get_rank
from panoptic_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Panoptic Segmentation Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["TEST.IMS_PER_BATCH","1"])
    cfg.freeze()

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    for data_loader_val in data_loaders_val:
        if cfg.MODEL.SEMANTIC.COMBINE_METHOD == "RAP":
            inference_relation(
                cfg,
                model,
                data_loader_val,
                device=cfg.MODEL.DEVICE,
            )
        elif cfg.MODEL.SEMANTIC.COMBINE_METHOD == "ocfusion":
            inference_relation_ocfusion(
                cfg,
                model,
                data_loader_val,
                device=cfg.MODEL.DEVICE,
            )
        synchronize()


if __name__ == "__main__":
    main()
