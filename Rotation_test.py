# coding=utf-8

# Author: Zylo117

"""
Simple Inference Rotation Visualize of EfficientDet-D2-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone
import os
import cv2
import numpy as np
from efficientdet.Rotation_utils import Rotation_BBoxTransform, ClipBoxes, BBoxAddScores
from utils.Rotation_utils import eval_preprocess ,preprocess ,  invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 2
force_input_size = None  

img_path = '/home/jobe/Rotated_airplane/images/P2214__0.7__0___0_h.jpg'

# print(img_path)
# print(torch.cuda.is_available())


anchor_ratios = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0) ] 
anchor_scales = [2 ** (0) , 2 ** (1.0 / 2.0)]
anchors_rotations = [ -90.0 ,-75.0 ,-60.0, -45.0, -30.0, -15.0 ]



threshold = 0.6
iou_threshold = 0.25

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True


obj_list = ['plane']

color_list = standard_to_bgr(STANDARD_COLORS)

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

ori_imgs, framed_imgs, framed_metas = eval_preprocess(img_path, max_size=input_size)


if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda()     for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi)            for fi in framed_imgs], 0)


x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone( compound_coef=compound_coef,  num_classes=len(obj_list),  ratios=anchor_ratios,  scales=anchor_scales , rotations= anchors_rotations )

model.load_state_dict(torch.load(f'/home/jobe/Rotated_airplane/logs/DOTA_effdet/efficientdet-d2_21_161605.pth', map_location='cpu'))
model.requires_grad_(False)

model.eval()

if use_cuda:
    model = model.cuda()

if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = Rotation_BBoxTransform()
    clipBoxes = ClipBoxes()
    addBoxes = BBoxAddScores()
    out = postprocess( x,
                       anchors, regression, classification,
                       regressBoxes, clipBoxes, addBoxes,
                       threshold, iou_threshold
                      )

def OPENCV2xywh(opencv_list):
    poly_list = []
    opencv_list[:5] = map(float, opencv_list[:5])
    x_c = int((opencv_list[0] + opencv_list[2]) / 2.)
    y_c = int((opencv_list[1] + opencv_list[3]) / 2.)
    width = int(opencv_list[2] - opencv_list[0])
    height = int(opencv_list[3] - opencv_list[1])
    theta = int(opencv_list[4])
    rect = ((x_c, y_c), (width, height), theta)
    poly = np.int0(cv2.boxPoints(rect))
    poly_list.append(poly)
    return poly_list 


def display(preds, imgs, imshow=False, imwrite=True):

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            """
            preds[i]['rois'][j] = [xmin, ymin, xmax, ymax, theta]
            
            """
            xmin, ymin, xmax, ymax, theta = preds[i]['rois'][j].astype(np.float64)

            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            if theta <= -90:
                color = [0, 255, 0]
            else:
                color = [255, 0, 0]
            rect = OPENCV2xywh([xmin, ymin, xmax, ymax, theta])
            j , k = rect[0][0 ,0] ,rect[0][0 ,1]

            cv2.drawContours(
                image=imgs[i],
                contours=rect,
                contourIdx=-1,
                color=color,
                thickness=2
                           )

        if imshow:
            cv2.imshow('immmmmmmmmmmmmmmmmmmmmmg', imgs[i])
            cv2.waitKey(0)
        
        if imwrite:
            path = '/home/jobe/Documents/Rotated_airplane/test'
            cv2.putText(imgs[i]  , f'{obj} ,  score:{score}' , org =(j, k) , fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.3, color=(0 , 0 ,0) ,thickness=1)
            cv2.imwrite(os.path.join(path , 'test_22.jpg')  ,imgs[i])
        

out = invert_affine(framed_metas, out)
display(out, ori_imgs, imshow=False, imwrite=True)

if __name__== '__main__' :
    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=True)
