# coding=utf-8

import os
import cv2
import numpy as np
import math
from tqdm import tqdm
import random
from collections import defaultdict
import argparse
import std_augmentation 


def make_dirs(paths):
    os.makedirs(os.path.join(paths, 'images'), exist_ok=True)
    os.makedirs(os.path.join(paths, 'labelTxt'), exist_ok=True)


def OPENCV2xywh(opencv_list):
    poly_list = []
    for idx in range(len(opencv_list)):
        
        x_c, y_c = int(opencv_list[0]), int(opencv_list[1])
        width, height = int(opencv_list[2]), int(opencv_list[3])
        theta = int(opencv_list[4])
        rect = ((x_c, y_c), (width, height), theta)
        poly = np.float32(cv2.boxPoints(rect))
        poly_list.append(poly)
    return poly_list


def VisualizeOPENCVformat(root_path, output_path):

    make_dirs(output_path)
    for visualize_name in tqdm(os.listdir(os.path.join(root_path, 'images')), ncols=88):
        if len (visualize_name.split('.')) == 2:
            basename_visualize_name = visualize_name.split('.')[0]
            
        else:
            basename_visualize_name = str(visualize_name.split('.')[0]) + '.' + str(visualize_name.split('.')[1])
           
        img = cv2.imread(os.path.join(root_path, 'images', basename_visualize_name + '.jpg'))
        with open(os.path.join(root_path, 'labelTxt', basename_visualize_name + '.txt'), 'r') as f_in:
            lines = f_in.readlines()
            
            head = []
            body = lines
            content = []
            for line in body:
                line = line.strip().split(' ')
                content.append(line)

            modified_list, zero_idx, positive_idx, vertical_idx = modify(content)

            if len(zero_idx) + len(positive_idx) == 0:
               
            else:
                temp_zero_list = []
                for idx in zero_idx:
                    temp_zero_list.append(content[idx])

            rect = OPENCV2xywh(content)
            rect = np.int0(rect)
            rect = np.array(rect)

            cv2.drawContours(image=img,
                             contours=rect,
                             contourIdx=-1,
                             color=[0, 0, 255],
                             thickness=2)
            cv2.imwrite(os.path.join(output_path, 'images/', f'{basename_visualize_name}.jpg'), img)


def write_img_label_Augment(args, lists):
 
    head = []
    format = ['vertical', 'horizontal', 'rotation']
    for idx in range(len(lists)):
        trans_foramt = format[idx]
        format_dict = lists[idx]

        for i in range(len(format_dict)):
            single_format_dict = format_dict[i]
            write_line_list = single_format_dict[0]['poly']

            img = single_format_dict[0]['trans_img']
            if trans_foramt == 'vertical':
                out_root_path = args.Aug_v_path
                new_txt_name = single_format_dict[0]['base_name'] + '_v' + '.txt'
                new_img_name = single_format_dict[0]['base_name'] + '_v' + '.jpg'
            if trans_foramt == 'horizontal':
                out_root_path =args.Aug_h_path
                new_txt_name = single_format_dict[0]['base_name'] + '_h' + '.txt'
                new_img_name = single_format_dict[0]['base_name'] + '_h' + '.jpg'
            if trans_foramt == 'rotation':
                out_root_path = args.Aug_t_path
                new_txt_name = single_format_dict[0]['base_name'] + '_r' + '.txt'
                new_img_name = single_format_dict[0]['base_name'] + '_r' + '.jpg'

            cv2.imwrite(os.path.join(args.out_root_path, out_root_path, 'images/', new_img_name), img)
            with open(os.path.join(args.out_root_path, out_root_path, 'labelTxt/', new_txt_name), 'w') as f_out:
                if len(head) != 0:
                    for j in range(len(head)):
                        f_out.write(head[j] + '\n')

                for i in range(len(write_line_list)):
                    line_content = write_line_list[i]
                    out_line = str(line_content[0]) + ' ' + str(line_content[1]) + ' ' + str(line_content[2]) \
                               + ' ' + str(line_content[3]) + ' ' + str(line_content[4]) + ' ' + str(line_content[5]) + ' ' + \
                               str(line_content[6]) + ' ' + str(line_content[7]) + ' ' + str(line_content[8]) + ' ' + str(line_content[9]) 
                    f_out.write(out_line + '\n')



def to_make_dirs(args):
    Aug_h_path_root = os.path.join(args.out_root_path, args.Aug_h_path)
    Aug_v_path_root = os.path.join(args.out_root_path, args.Aug_v_path)
    Aug_t_path_root = os.path.join(args.out_root_path, args.Aug_t_path)

    make_dirs(Aug_h_path_root)
    make_dirs(Aug_v_path_root)
    make_dirs(Aug_t_path_root)


def Rotate_vertex(poly, Pi_angle, a, b):
    X0 = (poly[0][0] - a) * math.cos(Pi_angle) - (poly[0][1] - b) * math.sin(Pi_angle) + a
    Y0 = (poly[0][0] - a) * math.sin(Pi_angle) + (poly[0][1] - b) * math.cos(Pi_angle) + b

    X1 = (poly[1][0] - a) * math.cos(Pi_angle) - (poly[1][1] - b) * math.sin(Pi_angle) + a
    Y1 = (poly[1][0] - a) * math.sin(Pi_angle) + (poly[1][1] - b) * math.cos(Pi_angle) + b

    X2 = (poly[2][0] - a) * math.cos(Pi_angle) - (poly[2][1] - b) * math.sin(Pi_angle) + a
    Y2 = (poly[2][0] - a) * math.sin(Pi_angle) + (poly[2][1] - b) * math.cos(Pi_angle) + b

    X3 = (poly[3][0] - a) * math.cos(Pi_angle) - (poly[3][1] - b) * math.sin(Pi_angle) + a
    Y3 = (poly[3][0] - a) * math.sin(Pi_angle) + (poly[3][1] - b) * math.cos(Pi_angle) + b

    return np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])


def Argument(args, ori_lists, is_horizontal=None, is_vertical=None, is_rotation=None):

    dict_horizontal = defaultdict(list)
    dict_vertical = defaultdict(list)
    dict_rotation = defaultdict(list)

    for idx in range(len(ori_lists)):
        single_dict = ori_lists[idx]  
        base_name = single_dict['basename']
        img = cv2.imread(os.path.join(args.root_path, 'images', base_name + '.jpg'))
        image_distort = std_augmentation.PhotometricDistort()

        width, height = single_dict['shape']['width'], single_dict['shape']['height']  # width, height of image
        horizontal_list = []
        if is_horizontal:
            horizontal = {}
            img_h = cv2.flip(img, 1)

            for i in range(len(single_dict['poly'])):
                single_obj = single_dict['poly'][i]
                x_c, y_c = float(single_obj[0]), float(single_obj[1])
                obj_w, obj_h = float(single_obj[2]), float(single_obj[3])
                obj_theta = float(single_obj[4])
                cls = single_obj[5]
                diff = single_obj[6]

                h_x_c = width - x_c
                h_y_c = y_c
                if obj_theta == -90:
                    h_theta = obj_theta
                    h_obj_h = obj_h
                    h_obj_w = obj_w

                else:
                    h_theta = float((90 - abs(obj_theta)) )
                    h_obj_w = obj_h
                    h_obj_h = obj_w
                     
                rect =OPENCV2xywh([h_x_c, h_y_c, h_obj_w, h_obj_h, h_theta])
                rect = np.int0(rect)
                rect = rect[0]
                
                    

                horizontal_list.append([ rect[0][0] , rect[0][1], rect[1][0] , rect[1][1] , rect[2][0], rect[2][1] , rect[3][0], rect[3][1], cls, diff ])
                
            horizontal['base_name'] = base_name
            horizontal['poly'] = horizontal_list
            horizontal['trans_img'] = img_h
            dict_horizontal[idx].append(horizontal)

        vertical_list = []
        if is_vertical:
            vertical = {}
            img_v = cv2.flip(img, 0)
            
            for i in range(len(single_dict['poly'])):
                single_obj = single_dict['poly'][i]
                x_c, y_c = float(single_obj[0]), float(single_obj[1])
                obj_w, obj_h = float(single_obj[2]), float(single_obj[3])
                obj_theta = float(single_obj[4])
                cls = single_obj[5]
                diff = single_obj[6]

                v_x_c, v_y_c = x_c, height - y_c

                if obj_theta == -90:
                    v_theta = obj_theta
                    v_obj_w = obj_w
                    v_obj_h = obj_h

                else:
                    v_theta = float((90 - abs(obj_theta)) )
                    v_obj_w = obj_h
                    v_obj_h = obj_w
                rect =OPENCV2xywh([h_x_c, h_y_c, h_obj_w, h_obj_h, h_theta])
                rect = np.int0(rect)
                rect = rect[0]


                vertical_list.append([rect[0][0] , rect[0][1], rect[1][0] , rect[1][1] , rect[2][0], rect[2][1] , rect[3][0], rect[3][1], cls, diff])
            vertical['base_name'] = base_name
            vertical['poly'] = vertical_list
            vertical['trans_img'] = img_v
            dict_vertical[idx].append(vertical)

        theta_list = []
        if is_rotation:
            rotation = {}
            base_degree = 45
            rotate_angle = random.uniform(-base_degree, base_degree) + 10
            print(f'line208:{rotate_angle}')
            radin_angle = -rotate_angle * math.pi / 180.0
            rotation_center_x, rotation_center_y = width / 2, height / 2
            M = cv2.getRotationMatrix2D(center=(rotation_center_x, rotation_center_y), angle=rotate_angle, scale=1)
            rotated_img = cv2.warpAffine(img, M, (width, height))
        

            for i in range(len(single_dict['poly'])):
                single_obj = single_dict['poly'][i]
                x_c, y_c = float(single_obj[0]), float(single_obj[1])
                obj_w, obj_h = float(single_obj[2]), float(single_obj[3])
                obj_theta = float(single_obj[4])
                cls = single_obj[5]
                diff = single_obj[6]

              
                rect = ((x_c, y_c), (obj_w, obj_h), obj_theta)
              
                vertex = cv2.boxPoints(rect)  
                vertex_arr = Rotate_vertex(vertex, radin_angle, rotation_center_x, rotation_center_y)
                rect_rotated = cv2.minAreaRect(np.float32(vertex_arr))

                rotation_c_x = rect_rotated[0][0]
                rotation_c_y = rect_rotated[0][1]
                rotation_w = rect_rotated[1][0]
                rotation_y = rect_rotated[1][1]
                rotation_theta = rect_rotated[-1]
                theta_list.append([rotation_c_x, rotation_c_y, rotation_w, rotation_y, rotation_theta, cls, diff])
                rotation['base_name'] = base_name
                rotation['poly'] = theta_list
                rotation['trans_img'] = rotated_img
                dict_rotation[idx].append(rotation)

    return dict_vertical, dict_horizontal, dict_rotation


def load_anno_and_make_new_anno(args):
    txt_path = os.path.join(args.root_path, 'labelTxt')
    img_path = os.path.join(args.root_path, 'images')

    txt_list = os.listdir(txt_path)
 
    anno_list = []

    for idx in range(len(txt_list)):
        whole_dict = {}
        shape_dict = {}
        basename = os.path.splitext(os.path.basename(txt_list[idx]))[0]
        img_name = basename + '.jpg'
        txt_name = basename + '.txt'

        
        img = cv2.imread(os.path.join(img_path, img_name))
        height, width, _ = img.shape
        shape_dict['width'] = width
        shape_dict['height'] = height
        whole_dict['basename'] = basename
        whole_dict['shape'] = shape_dict

        with open(os.path.join(txt_path, txt_name)) as f_in:
            lines = f_in.readlines()
         
            body = lines[:]
            file_content = []
            for line in body:
                line = line.strip().split(' ')
                file_content.append(line)
            whole_dict['poly'] = file_content  
        anno_list.append(whole_dict)
    return anno_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default=r'/home/jobe/Documents/Rotated_airplane/conte/test')

    parser.add_argument('--Aug_h_path', type=str,
                        default=r'Augment_horizontal/')

    parser.add_argument('--Aug_v_path', type=str,
                        default=r'Augment_vertical/')

    parser.add_argument('--Aug_t_path', type=str,
                        default=r'Augment_rotation/')

    parser.add_argument('--out_root_path', type=str,
                        default=r'/home/jobe/Documents/Rotated_airplane/conte/augg/')

    args = parser.parse_args()

    to_make_dirs(args)
    
    anno_lists = load_anno_and_make_new_anno(args)
  
    vertical_dict, horizontal_dict, rotational_dict = Argument(args, anno_lists, is_horizontal= True , is_vertical=  True, is_rotation=False )
    
    write_img_label_Augment(args, [vertical_dict, horizontal_dict, rotational_dict])


    # VisualizeOPENCVformat( root_path=r'/home/jobe/Rotation_Effdet_768/datasets/Augment_rotation/', 
    #                                    output_path='/home/jobe/Rotation_Effdet_768/opencv')


