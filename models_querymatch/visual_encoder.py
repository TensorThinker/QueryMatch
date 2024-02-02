# coding=utf-8
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from utils_querymatch.distributed import seed_everything

backbone_dict={
    'mask2former': build_model
}


def visual_encoder(__C, cfg):
    seed_everything(__C.SEED)
    vis_enc=backbone_dict[__C.VIS_ENC](cfg)
    vis_enc.eval()
    checkpointer = DetectionCheckpointer(vis_enc)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return vis_enc

