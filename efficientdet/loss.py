
# coding=utf-8
import torch
import torch.nn as nn
import cv2
import numpy as np
import math
from random import sample
from efficientdet.polyiou import polyiou
from decimal import Decimal
import matplotlib.pyplot as plt



def find_index(lists):
    result_list = []
    for idx in range(len(lists)):
        if lists[idx] == True:
            result_list.append(idx)
    return result_list


def visualize_Rectangle_area(image_path, lists):
    image = cv2.imread(image_path)
    rect = np.array(np.int0(lists))
    cv2.drawContours(image=image,
                     contours=rect,
                     contourIdx=-1,
                     color=[0, 0, 255],
                     thickness=2)
    cv2.imwrite('Rectangle.jpg', image)


def visualize_poly2Horizontal_rect(image_path, lists):
    img = cv2.imread(image_path)
    for idx in range(len(lists)):
        lists[idx][:4] = map(int, lists[idx][:4])
        xmin, ymin, xmax, ymax = lists[idx][0], lists[idx][1], lists[idx][2], lists[idx][3]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=[0, 255, 0], thickness=1)
    cv2.imwrite('rect.jpg', img)


def visualize_positive_anchor(image_path, anchor_lists, index_lists):
    anchor_lists = anchor_lists.cpu().numpy().tolist()
    image = cv2.imread(image_path)
    sample_list = sample(index_lists, len(index_lists))
    for idx in sample_list:
        single_anchor = anchor_lists[idx]
        single_anchor[:4] = map(int, single_anchor[:4])
        x1, y1, x2, y2 = single_anchor[0], single_anchor[1], single_anchor[2], single_anchor[3]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

    cv2.imwrite('anchor.jpg', image)


def visualize_rp_anchor(path, lists):
    image = cv2.imread(path)
    if torch.is_tensor(lists):
        lists = lists.cpu().numpy().tolist()

    for idx in range(len(lists)):
        lists[idx][:4] = map(int, lists[idx][:4])
        xlt, ylt, xrb, yrb = lists[idx][0], lists[idx][1], lists[idx][2], lists[idx][3]
        cv2.rectangle(image, (xlt, ylt), (xrb, yrb), color=[255, 0, 0], thickness=1)
    cv2.imwrite('anchor.jpg', image)


def check_anchor(path, lists):
    image = cv2.imread(path)
    lists = lists.cpu().numpy().tolist()
    for idx in range(len(lists)):
        single_list = lists[idx]
        single_list[:] = map(int, single_list[:])
        xmin, ymin, xmax, ymax = single_list[0], single_list[1], single_list[4], single_list[5]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=[255, 255, 0], thickness=1)
    cv2.imwrite('anchor.jpg', image)


def visualize_pp_anchor(path, lists):
    image = cv2.imread(path)
    if torch.is_tensor(lists):
        lists = lists.cpu().numpy().tolist()
    for idx in range(len(lists)):
        templist = list(map(int, lists[idx][:4]))
        xmin, ymin, xmax, ymax = templist[0], templist[1], templist[2], templist[3]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=[0, 240, 0], thickness=1)
    cv2.imwrite('anchor.jpg', image)


def Rectangle_area(rotation_bbox):
    poly_box = []
    for i in range(len(rotation_bbox)):
        single_box = rotation_bbox[i]
        x_c, y_c = int(single_box[0]), int(single_box[1])
        width, height = int(single_box[2]), int(single_box[3])
        theta = int(single_box[4])
        rect = ((x_c, y_c), (width, height), theta)
        poly = np.int0(cv2.boxPoints(rect)) 
        poly_box.append(poly)
    return poly_box


def single_Rectangle_area(rotation_bbox):
    x_c, y_c = int(rotation_bbox[0]), int(rotation_bbox[1])
    width, height = int(rotation_bbox[2]), int(rotation_bbox[3])
    theta = int(rotation_bbox[4])
    rect = ((x_c, y_c), (width, height), theta)
    poly = np.int0(cv2.boxPoints(rect))
    return poly


def Rotation2points(Poly):
    lists = []
    for idx in range(len(Poly)):
        lists.append(Poly[idx][0])
        lists.append(Poly[idx][1])
    return lists


def poly2Horizontal_rect(poly):
    total_list = []
    for idx in range(len(poly)):
        xlist = []
        ylist = []
        for j in range(len(poly[idx])):
            xlist.append(poly[idx][j][0])
            ylist.append(poly[idx][j][1])
        xmin = np.min(xlist)
        xmax = np.max(xlist)
        ymin = np.min(ylist)
        ymax = np.max(ylist)
        total_list.append([xmin, ymin, xmax, ymax])
    return total_list


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    horizontal_IoU = intersection / ua
    return horizontal_IoU

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :] 
        # anchor --> [xlt, ylt, xrb, yrb, theta]
        dtype = anchors.dtype

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        anchor_theta = anchor[:, 4]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
           
            bbox_annotation = bbox_annotation[bbox_annotation[:, 5] != -1]
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue


            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            vertex = Rectangle_area(bbox_annotation[:, :5])  
            horizontal_vertex = poly2Horizontal_rect(vertex)  

            if torch.cuda.is_available():
                horizontal_vertex = torch.tensor(horizontal_vertex).cuda()

            HIoU = calc_iou(anchor[:, :], horizontal_vertex)  
            hor_IoU_max, hor_IoU_argmax = torch.max(HIoU, dim=1)

         
            hor_positive_indices = torch.ge(hor_IoU_max, 0.6)

            parent_num_list = np.arange(len(anchor))
            parent_positive_index = list(parent_num_list[hor_positive_indices.cpu().numpy()])

            positive_anchor_list = anchor[hor_positive_indices, :]

            assigned_annotations = bbox_annotation[hor_IoU_argmax, :]

            hor_positive_assigned_annotations = assigned_annotations[hor_positive_indices, :]

            anchor_widths_pi = anchor_widths[hor_positive_indices]  # 
            anchor_heights_pi = anchor_heights[hor_positive_indices]
            anchor_ctr_x_pi = anchor_ctr_x[hor_positive_indices]
            anchor_ctr_y_pi = anchor_ctr_y[hor_positive_indices]

            xlt, ylt = anchor_ctr_x_pi - anchor_widths_pi / 2, anchor_ctr_y_pi - anchor_heights_pi / 2
            xrt, yrt = anchor_ctr_x_pi + anchor_widths_pi / 2, anchor_ctr_y_pi - anchor_heights_pi / 2
            xrb, yrb = anchor_ctr_x_pi + anchor_widths_pi / 2, anchor_ctr_y_pi + anchor_heights_pi / 2
            xlb, ylb = anchor_ctr_x_pi - anchor_widths_pi / 2, anchor_ctr_y_pi + anchor_heights_pi / 2

            anchor_vertex = torch.stack([xlt, ylt, xrt, yrt, xrb, yrb, xlb, ylb]).t()
            
            skew_IoU_lists = []

            for index in range(len(anchor_vertex)):
                single_anchor = anchor_vertex[index]
                single_rotation_opencv_format = hor_positive_assigned_annotations[index][:5]
                single_rotation_box = single_Rectangle_area(single_rotation_opencv_format)
                single_points = Rotation2points(single_rotation_box)


                result1 = list(map(lambda x: float(x), single_points))
                result2 = list(map(lambda x: float(x), single_anchor))

                skew_IoU = polyiou.iou_poly(polyiou.VectorDouble(result1), polyiou.VectorDouble(result2))
                skew_IoU = np.array(skew_IoU).astype(np.float64)
                skew_IoU_lists.append(skew_IoU)

            skew_IoU_lists = np.array(skew_IoU_lists)

            if not torch.is_tensor(skew_IoU_lists):
                overlaps = torch.from_numpy(skew_IoU_lists).cuda(0)
       
            rotation_threshold = 0.3
            rotation_positive_indices = torch.ge(overlaps, rotation_threshold)


            son_num_list = np.arange(len(overlaps))
            son_positive_index = list(son_num_list[rotation_positive_indices.cpu().numpy()])  # 

         
            targets[torch.lt(hor_IoU_max, 0.4), :] = 0

        
            num_positive_anchors = rotation_positive_indices.sum()

            
            positive_index_list = np.zeros(len(anchor), dtype=bool)
            for idx in range(len(son_positive_index)):
                son_idx = son_positive_index[idx]
                parent_idx = parent_positive_index[son_idx]
                positive_index_list[parent_idx] = 1

            targets[positive_index_list, :] = 0
            targets[positive_index_list, assigned_annotations[positive_index_list, 5].long()] = 1


            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            # Zylo code
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)

            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
          
            if rotation_positive_indices.sum() > 0:

                # GT 
                rotation_assigned_annotations_gt = hor_positive_assigned_annotations[rotation_positive_indices]

                
                rotation_anchor_widths_pi = anchor_widths[positive_index_list]
                rotation_anchor_heights_pi = anchor_heights[positive_index_list]
                rotation_anchor_ctr_x_pi = anchor_ctr_x[positive_index_list]
                rotation_anchor_ctr_y_pi = anchor_ctr_y[positive_index_list]
                rotation_anchor_theta = anchor_theta[positive_index_list]


                # efficientdet style
                rotation_assigned_annotations_gt[:, 2] = torch.clamp(rotation_assigned_annotations_gt[:, 2], min=1)
                rotation_assigned_annotations_gt[:, 3] = torch.clamp(rotation_assigned_annotations_gt[:, 3], min=1)

                targets_dx = (rotation_assigned_annotations_gt[:, 0] - rotation_anchor_ctr_x_pi) / rotation_anchor_widths_pi
                targets_dy = (rotation_assigned_annotations_gt[:, 1] - rotation_anchor_ctr_y_pi) / rotation_anchor_heights_pi
                targets_dw = torch.log(rotation_assigned_annotations_gt[:, 2] / rotation_anchor_widths_pi)
                targets_dh = torch.log(rotation_assigned_annotations_gt[:, 3] / rotation_anchor_heights_pi)
                targets_theta = ((rotation_assigned_annotations_gt[:, 4] / 180 * math.pi) - (rotation_anchor_theta / 180 * math.pi))

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_theta))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_index_list, :])

                # smooth l1 loss
                regression_loss = torch.where(
                    torch.le(regression_diff, 1),
                    0.5 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5

                )

                regression_losses.append(regression_loss.mean())

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0,
                                                   keepdim=True) *50


if __name__ == '__main__':
    

    from torchvision import transforms
    import os
    from torch.utils.data import DataLoader
    import argparse
    import yaml

    from efficientdet.utils import Anchors

    from efficientdet.rotation_dataset import RotationCocoDataset, collater
    from efficientdet.rotation_dataset import Normalizer, Resizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default=r'Rotation-EfficinetDet/datasets/')
    args = parser.parse_args()

    yaml_rootpath = r'Rotation-EfficinetDet/projects/'
    yamlpath = os.path.join(yaml_rootpath, 'dataset.yml')


    class Params:
        def __init__(self, project_file):
            self.params = yaml.safe_load(open(project_file).read())

        def __getattr__(self, item):
            return self.params.get(item, None)


    params = Params(yamlpath)
    test_batch_size = 1
    training_params = {'batch_size': test_batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': 1}

    training_set = RotationCocoDataset(
        root_dir=os.path.join(args.root_path, params.project_name),
        set=params.train_set,
        transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                      Resizer(512)]))

    training_generator = DataLoader(training_set, **training_params)
    num_iter = len(training_generator)  

    dataiter = iter(training_generator)
    iter_content = dataiter.next()

    annot = iter_content['annot']
    annotation = annot.cuda(0)

    img = iter_content['img']
    images = img.cuda(0)

    anchor = Anchors()
    anchors = anchor.forward(images) 

    FL = FocalLoss()
    classifications = torch.from_numpy(np.ones([test_batch_size, 49104, 2]) * 0.5).cuda(0)
    regressions = torch.from_numpy(np.ones([test_batch_size, 49104, 5]) * 0.5).cuda(0)
    FL.forward(classifications=classifications, regressions=regressions, anchors=anchors, annotations=annotation)


    
