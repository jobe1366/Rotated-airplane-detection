# coding=utf-8

# Author: Zylo117





"""
Simple Inference Rotation Visualize of EfficientDet
"""
import time
import torch
from torch.backends import cudnn
from tqdm import tqdm
from matplotlib import colors
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone
import cv2
import numpy as np
from efficientdet.Rotation_utils import Rotation_BBoxTransform, ClipBoxes, BBoxAddScores
from utils.Rotation_utils import eval_preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os


def OPENCV2xywh(opencv_list):
    poly_list = []
    opencv_list[:5] = map(float, opencv_list[:5])
    x_c = int((opencv_list[0] + opencv_list[2]) / 2.)
    y_c = int((opencv_list[1] + opencv_list[3]) / 2.)
    width = int(opencv_list[2] - opencv_list[0])
    height = int(opencv_list[3] - opencv_list[1])
    theta = int(opencv_list[4])
    rect = ((x_c, y_c), (width, height), theta)
    poly = np.float32(cv2.boxPoints(rect))
    poly_list.append(poly)
    return poly_list


def write_into_txt(file_name, lists):
    path = r'/home/jobe/Rotation_Effdet/result _for_each_class'
    for idx in range(len(lists)):
        single_list = lists[idx]
        class_id = single_list[0]
        txt_name = file_name[class_id]
        txt_path = os.path.join(path, txt_name)

        with open(txt_path, 'a') as f_out:
            strline = str(single_list[1]) + ' ' + str(single_list[2]) + ' ' + str(single_list[3]) + \
                      ' ' + str(single_list[4]) + ' ' + str(single_list[5]) + ' ' + str(single_list[6]) + \
                      ' ' + str(single_list[7]) + ' ' + str(single_list[8]) + ' ' + str(single_list[9]) + \
                      ' ' + str(single_list[10]) + '\n'
            f_out.write(strline)


compound_coef = 2
force_input_size = None  

anchor_ratios = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0) ] 
anchor_scales = [2 ** (0) , 2 ** (1.0 / 2.0)]
anchors_rotations = [ -90.0 ,-75.0 ,-60.0, -45.0, -30.0, -15.0 ]

threshold = 0.6
iou_threshold = 0.2
use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['plane']


color_list = standard_to_bgr(STANDARD_COLORS)
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)


model.load_state_dict(torch.load(f'/home/jobe/Rotation_Effdet/logs/DOTA_effdet/efficientdet-d3_21_118000.pth',
                                 map_location='cpu'))


model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()

if use_float16:
    model = model.half()



path = r'imgfile.txt'
imgpath = r'/EfficientDet/images'

content = []
with open(path, 'r') as f_in:
    lines = f_in.readlines()
    for idx in range(len(lines)):
        line = lines[idx]
        line = line.strip().split(' ')
        content.append(line[0])



for i in tqdm(range(len(content)), ncols=88):
    filebasename = content[i]
    img_path = os.path.join(imgpath, filebasename + '.jpg')

    ori_imgs, framed_imgs, framed_metas = eval_preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)


    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = Rotation_BBoxTransform()
        clipBoxes = ClipBoxes()
        addBoxes = BBoxAddScores()
        
        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes, addBoxes,
                          threshold, iou_threshold)


    def OPENCV2xywh(opencv_list):
        poly_list = []
        for idx in range(len(opencv_list)):
            opencv_list[idx][:5] = map(float, opencv_list[idx][:5])
            x_c, y_c = int(opencv_list[idx][0]), int(opencv_list[idx][1])
            width, height = int(opencv_list[idx][2]), int(opencv_list[idx][3])
            theta = int(opencv_list[idx][4])
            rect = ((x_c, y_c), (width, height), theta)
            poly = np.float32(cv2.boxPoints(rect))
            poly_list.append(poly)
        return poly_list


    def display(filebasenme, preds, imgs, imshow=True, imwrite=False):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            imgs[i] = imgs[i].copy()

            for j in range(len(preds[i]['rois'])):
                """
                preds[i]['rois'][j] = [xmin, ymin, xmax, ymax, theta]
                """
                xmin, ymin, xmax, ymax, theta = preds[i]['rois'][j].astype(np.float)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                
                if theta <= -90:
                    color = [0, 255, 0]
                else:
                    color = [0, 0, 255]
               
                rect = OPENCV2xywh([xmin, ymin, xmax, ymax, theta])
                rect = np.int0(rect)
                rect = np.array(rect)

              
                cv2.drawContours(
                    image=imgs[i],
                    contours=rect,
                    contourIdx=-1,
                    color=color,
                    thickness=2
                )

            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)


            if imwrite:
                cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{filebasename}.jpg', imgs[i])


    out = invert_affine(framed_metas, out)
    file_name = ['Task1_large-vehicle.txt', 'Task1_small-vehicle.txt']
    rois = out[0]['rois']
    class_ids = out[0]['class_ids']
    scores = out[0]['scores']


    filecontent = []
    for ii in range(len(scores)):
        xmin, ymin, xmax, ymax, theta = rois[ii]
        rect = OPENCV2xywh([xmin, ymin, xmax, ymax, theta])[0].tolist()
        x1, y1 = float(rect[0][0]), float(rect[0][1])
        x2, y2 = float(rect[1][0]), float(rect[1][1])
        x3, y3 = float(rect[2][0]), float(rect[2][1])
        x4, y4 = float(rect[3][0]), float(rect[3][1])
        single_filecontent = [int(class_ids[ii]), filebasename, float(scores[ii]), x1, y1, x2, y2, x3, y3, x4, y4]
        filecontent.append(single_filecontent)

    write_into_txt(file_name, filecontent)


