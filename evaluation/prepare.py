"""Prepare to create imgnamefile.txt file and result_classnmae folder."""
# coding=utf-8

import json
import os


def load_json(path):
    with open(path, 'r') as f_in:
        content = json.load(f_in)
    fileimages = content['images']
    image_lists = []
    for idx in range(len(fileimages)):
        single_image = fileimages[idx]
        image_lists.append(os.path.splitext(single_image['file_name'])[0])

    return image_lists

def write_file(path, img_lists):
    print(len(img_lists))
    with open(path, 'w') as f_out:
        for idx in range(len(img_lists)):
            if idx < len(img_lists) - 1:
                f_out.write(img_lists[idx])
                f_out.write('\n')
            if idx == len(img_lists) - 1:
                f_out.write(img_lists[idx])


def check_and_write_anno_file(path, txt_path, lists):
    head = ['imagesource:GoogleEarth', 'gsd:0.115726939386']
    invalid = []
    for idx in range(len(lists)):
        flag = 1
        txtfile_name = os.path.join(path, lists[idx] + '.txt')
        with open(txtfile_name, 'r') as f_in:
            file_content = []
            lines = f_in.readlines()
            for i in range(len(lines)):
                line = lines[i]
                split_lines = line.strip().split(' ')
                if len(split_lines) < 10:
                    invalid.append(txtfile_name)
                    flag = 0
                    continue
                if split_lines[8] == 'plane' :
                       split_lines[:8] = map(lambda x: float(x), split_lines[:8])   
                       file_content.append(split_lines)
                else:
                    continue
            if flag == 1:
                with open(os.path.join(txt_path, lists[idx] + '.txt'), 'w') as f_out:
                    f_out.write(head[0] + '\n')
                    f_out.write(head[1] + '\n')
                    for i in range(len(file_content)):
                        strline = str(file_content[i][0]) + ' ' + str(file_content[i][1]) + ' ' + str(file_content[i][2]) + ' ' \
                        + str(file_content[i][3]) + ' ' + str(file_content[i][4]) + ' ' + str(file_content[i][5]) + ' ' \
                        + str(file_content[i][6]) + ' ' + str(file_content[i][7]) + ' ' + file_content[i][8] + ' ' + file_content[i][9]
                        f_out.write(strline)
                        if i < len(file_content) - 1:
                            f_out.write('\n')
    return invalid


if __name__ == '__main__':
    json_path = r'/home/jobe/Rotated_airplane/dataset/DOTA_effdet/annotations/instances_test.json'   
    image_lists = load_json(json_path)
    imgnamefile_path = r'/home/jobe/Rotated_airplane/evaluation/imgnamefile.txt'
    write_file(imgnamefile_path ,image_lists )
    print(f'{imgnamefile_path} created.')

    txt_path = r'/home/jobe/Documents/Rotated_airplane/conte/iiiiii/test1/labelTxt'             
    dst_path = r'/home/jobe/Documents/Rotated_airplane/evaluation/gt_labels'        
    
    invalid_list = check_and_write_anno_file(txt_path, dst_path, image_lists )

    # print(f"invalidddddd  format of labelssss?????? ：{ len (invalid_list)   }")
