# coding=utf-8

import numpy as np
from collections import defaultdict
import cv2


def record_offset(img_list):
    
    offset_list = defaultdict(list)
    for index in range(len(img_list)):
        img_name = img_list[index]
        w_off = img_name.split('_')[1]
        h_off = img_name.split('_')[2]
        print(f'w_off:{w_off}, h_off{h_off}')
        offset_list[index] = [h_off, w_off]
    return offset_list


def add_offset(offsets, bboxs):
 
    small_x1 = bboxs[0]
    small_y1 = bboxs[1]
    small_x2 = bboxs[2]
    small_y2 = bboxs[3]

    offset_w = int(offsets[1])
    offset_h = int(offsets[0])

    big_y1 = small_y1 + offset_h
    big_y2 = small_y2 + offset_h
    big_x1 = small_x1 + offset_w
    big_x2 = small_x2 + offset_w

    return [big_x1, big_y1, big_x2, big_y2]


def py_nms(dets, thresh=0.2):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、score

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

   
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep  


def plot_one_box_new(img, coord, label=None, score=None, color=[0, 255, 255], line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1) 
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
    return img


if __name__ == '__main__':
    img_list = ['P037.png']
    result = record_offset(img_list)
    print(result[0])
