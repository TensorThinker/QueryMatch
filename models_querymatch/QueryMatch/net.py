# coding=utf-8

import torch
import torch.nn as nn
from models_querymatch.language_encoder import language_encoder
from models_querymatch.visual_encoder import visual_encoder
from models_querymatch.QueryMatch.head import WeakREChead
from models_querymatch.network_blocks import MultiScaleFusion
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from torchvision import transforms
import numpy as np
from detectron2.structures import Boxes, ImageList, Instances
from utils_querymatch.utils import normed2original
from detectron2.data.detection_utils import read_image
import cv2
import torch.nn.functional as F

    
class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, cfg):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C, cfg).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)
        self.linear_vs = nn.Linear(256, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.head = WeakREChead(__C)
        self.device = cfg.MODEL.DEVICE
        self.class_num = __C.CLASS_NUM
        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)
        self.cfg = cfg
        self.__C = __C

    def preprocess_img(self,img):
        h, w, _ = img.shape
        imgsize=self.__C.INPUT_SHAPE[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2
        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        return sized


    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x, y, input_shape,ids=None,stat_sim_dict=None):
        with torch.no_grad():
            outputs = self.visual_encoder(x,self.device,input_shape)  
        decoder_output = outputs["decoder_outputs"]  
        instances = outputs["instances"] 
        all_instance = instances.detach().clone()
        instances_scores = outputs["instances_scores"]  
        y_ = self.lang_encoder(y)
        vals, indices = instances_scores.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, insnum = instances.shape[:2]
        instances = instances.masked_select(
            torch.zeros(bs, insnum).to(self.device).scatter(1,indices,1).bool().unsqueeze(2).unsqueeze(
                3).expand(bs, insnum, instances.shape[-2], instances.shape[-1])).contiguous().view(bs, self.select_num, instances.shape[-2], instances.shape[-1])
        decoder_output = decoder_output.masked_select(
            torch.zeros(bs, insnum).to(self.device).scatter(1,indices,1).bool().unsqueeze(2).expand(bs, insnum, decoder_output.shape[-1])).contiguous().view(bs, self.select_num, decoder_output.shape[-1])
        x_new = self.linear_vs(decoder_output)
        lan_encoded = y_["flat_lang_feat"]
        y_new = self.linear_ts(lan_encoded.unsqueeze(1))
        if self.training:
            loss, stat_sim_dict = self.head(x_new, y_new, stat_sim_dict)  
            return loss, stat_sim_dict
        else:
            predictions_s = self.head(x_new, y_new)
            res_mask_idx = predictions_s.squeeze(1)
            instances = instances[res_mask_idx]
            return instances,all_instance