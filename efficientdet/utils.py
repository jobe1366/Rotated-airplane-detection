# coding=utf-8

import itertools
import torch
import torch.nn as nn
import numpy as np

###################################################################################################################33
class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa 
        h = regression[..., 2].exp() * ha  

        y_centers = regression[..., 0] * ha + y_centers_a  
        x_centers = regression[..., 1] * wa + x_centers_a  

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2) 

###########################################################################################################33333333
class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


####################################################################################################333

def generate_anchors(base_size, ratios, scales, rotations):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    num_anchors = len(ratios) * len(scales) * len(rotations)
    anchors = np.zeros((num_anchors, 5))
    anchors[:, 2:4] = base_size * np.tile(scales, (2, len(ratios) * len(rotations))).T
    for idx in range(len(ratios)):
        anchors[3 * idx: 3 * (idx + 1), 2] = anchors[3 * idx: 3 * (idx + 1), 2] * ratios[idx][0]
        anchors[3 * idx: 3 * (idx + 1), 3] = anchors[3 * idx: 3 * (idx + 1), 3] * ratios[idx][1]

    anchors[:, 4] = np.tile(np.repeat(rotations, len(scales)), (1, len(ratios))).T[:, 0]
    anchors[:, 0:3:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1:4:2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

########################################################################################################################

def shift(shape, stride, anchors):
    shift_x = np.arange(stride / 2, shape[1], stride)
    shift_y = np.arange(stride / 2, shape[0], stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel(),
        np.zeros(shift_x.ravel().shape)
    )).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]  
    all_anchors = (anchors.reshape((1, A, 5)) + shifts.reshape((1, K, 5)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 5))
    return all_anchors

############################################################################################################################
class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    2021/04/19 modified the function of the class Anchors to the Rotation Anchors
    """

    def __init__(self, anchor_scale= 2.0, pyramid_levels=None,  **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [  4, 5, 6, 7 ]  
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 2.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0),(2.0, 1.0),(1.0, 2.0) ] )
        self.rotations = np.array(kwargs.get('rotations' , [ -90.0 ,-75.0 ,-60.0, -45.0, -30.0, -15.0 ]))
        self.base_sizes = [x * anchor_scale for x in self.strides]
        self.last_anchors = {}
        self.last_shape = None
    

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale rotation anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          rotation_anchor_boxes: a numpy array with shape [N, 5](x1, y1, x2, y2, theta=0),
          which stacks anchors on all feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]  # image Tensor (B, C, H, W) 

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        all_anchors = np.zeros((0, 5)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            rotation_anchors = generate_anchors(
                base_size=self.base_sizes[idx],
                ratios=self.ratios,
                scales=self.scales,
                rotations=self.rotations
            )

            shifted_anchors = shift(image_shape, self.strides[idx], rotation_anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = np.tile(all_anchors, (image.shape[0], 1, 1))  
        all_anchors = torch.from_numpy(all_anchors.astype(dtype))
        if torch.is_tensor(image) and image.is_cuda:
            all_anchors = all_anchors.cuda()  
        return all_anchors

###########################################################################################################################

if __name__ == '__main__':
    import cv2
    import random
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
 

    img_arr = torch.Tensor(1, 3, 7876, 7876 ).cuda()
    image = cv2.imread(r'/home/jobe/Rotated_airplane/dataset/DOTA_effdet/val/images/P0023__0.6__0___1236.jpg')
    anchors = Anchors()
    Anchors_all_level = anchors.forward(img_arr)
    base_anchor = Anchors_all_level[:, 0:23, :].cpu().numpy() - np.array([2., 2., 2., 2., 0])
    classes = len(example[0])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(classes)]
    
    

    for idx in range(classes):
        single_example = example[0][idx]
        single_color = colors[idx]
        single_example[:4] = map(int, single_example[:4])
        xlt, ylt = single_example[0], single_example[1]
        xrb, yrb = single_example[2], single_example[3]
        rect = OPENCV2xywh(single_example)
        rect = np.array(np.int0(rect))

        cv2.drawContours(
             image=image,
             contours=rect,
             contourIdx=-1,
             color=single_color,
             thickness=2
         )
        cv2.imwrite('/home/jobe/Rotated_airplane/test/img44444_anchor44.jpg', image)
