"""
1-clip_images.py can clip one dataset into four folder with different density.
In every folder ,there are clipped images,clipped density maps and clipped ground truth.
So,this script is to divide three kinds of files above into three folders

"""

import sys
import os
import numpy as np
from multiprocessing import Process


def file_classify(clip_result_path, divided_path, file_list, folder_list, block_index):
    for k, file in enumerate(file_list):
        # 先把.npy结尾的放入ground_truth文件夹
        file_type = file.split('.')[-1]

        finished = k / len(file_list) * 100
        print('process {} finished {:.3f}%'.format(block_index + 1, finished))

        if file_type == 'npy':
            os.system('sudo cp {}/{} {}/{}'.format(clip_result_path, file, divided_path, folder_list[2]))
        elif 'density' in file.split('_'):
            os.system('sudo cp {}/{} {}/{}'.format(clip_result_path, file, divided_path, folder_list[1]))
        else:
            os.system('sudo cp {}/{} {}/{}'.format(clip_result_path, file, divided_path, folder_list[0]))


def divide_into_three_folders(clip_result_path, divided_path, num_process):
    clipped_images_folder = 'images'
    clipped_densitymap_folder = 'density_map'
    clipped_groundtruth_folder = 'ground_truth'

    folders = [clipped_images_folder, clipped_densitymap_folder, clipped_groundtruth_folder]

    # 先创建三个文件夹
    if not os.path.exists(divided_path):
        os.mkdir(divided_path)
    else:
        return 0
    for folder in folders:
        folder_path = divided_path + '/' + folder
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    # 使用多进程去处理
    total_file_list = os.listdir(clip_result_path)
    total_num = len(total_file_list)

    # 计算每个进程要处理的图片数量
    num_pic_per_process = round(total_num / num_process)
    num_pic_last_process = total_num - 7 * num_pic_per_process

    # 将文件分成8块
    fileblock_list = []
    for i in range(num_process):
        if i < num_process - 1:
            start_index = i * num_pic_per_process
            end_index = (i + 1) * num_pic_per_process
        else:
            start_index = i * num_pic_per_process
            end_index = start_index + num_pic_last_process

        file_block = total_file_list[start_index:end_index]
        fileblock_list.append(file_block)

    process = [Process(target=file_classify,
                       args=(clip_result_path, divided_path, block, folders, block_index))
               for block_index, block in enumerate(fileblock_list)]

    for p in process:
        p.start()
    for p in process:
        p.join()


"""--------------------------------------执行程序-------------------------------------------"""
# 配置clip之后的文件夹路径
num_process = 8
file_path = sys.argv[1]

density_list = ['XL_density', 'L_density', 'M_density', 'S_density']

clip_result_path_list = []
divided_path_list = []

for folder_name in density_list:
    clip_result_path = file_path + '/' + folder_name
    clip_result_path_list.append(clip_result_path)
    if not os.path.exists(clip_result_path):
        print('can not found {}'.format(file_path))
        exit(0)

    # 配置divided_path
    divided_path = ''
    for i in range(len(clip_result_path.split('/')) - 1):
        divided_path = divided_path + clip_result_path.split('/')[i] + '/'

    divided_path = divided_path + '{}_divided'.format(clip_result_path.split('/')[-1])

    print('dealing with {}...................'.format(clip_result_path))
    divide_into_three_folders(clip_result_path, divided_path, num_process)
