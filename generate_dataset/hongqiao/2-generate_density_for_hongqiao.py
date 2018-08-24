"""
this script is to generate density map for hongqiao dataset
"""
from data_process_before_train import generate_gaussian_kernel
import numpy as np
import cv2
import os
from xml.dom.minidom import parse

"""--------------------输入输出路径配置----------------------"""
# 需要调的参数就是高斯核sigma
sigma = 4
kernel_size = 15
max_kernel_size=100
fixed_kernel=False

gt_path = '../../../dataset/hongqiao/gt/gt'
density_path = '../../../dataset/hongqiao/gt/density_map'
density_image_path = '../../../dataset/hongqiao/gt/density_image'

if not os.path.exists(density_path):
    os.mkdir(density_path)
if not os.path.exists(density_image_path):
    os.mkdir(density_image_path)

"""--------------------将每一张gt转化为density_map----------------------"""
gt_list = os.listdir(gt_path)
gt_list.sort()

for i, gt in enumerate(gt_list):
    density_map = np.zeros((300, 480))
    print("process {}".format(gt))
    DOMTree = parse(gt_path + '/' + gt)
    xml_content = DOMTree.documentElement
    bndboxs = xml_content.getElementsByTagName('bndbox')

    # 对每个boundingbox进行高斯核填充
    for bndbox in bndboxs:
        col1 = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
        row1 = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
        col2 = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
        row2 = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)

        # 对坐标缩放
        col1 = round(col1 / 4)
        col2 = round(col2 / 4)
        row1 = round(row1 / 4)
        row2 = round(row2 / 4)

        # 计算矩形框中心
        center_row = round((row1 + row2) / 2)
        center_col = round((col1 + col2) / 2)

        # 如果变尺寸高斯核
        if not fixed_kernel:
            # 如果标的框尺寸没有超过max_kernel_size,就保留原尺寸，如果超过了就限制在最大值
            if row2 - row1 > max_kernel_size:
                row2 = center_row + int((max_kernel_size - 1) / 2)
                row1 = center_row - int((max_kernel_size - 1) / 2)
            if col2 - col1 > max_kernel_size:
                col2 = center_col + int((max_kernel_size - 1) / 2)
                col1 = center_col - int((max_kernel_size - 1) / 2)
        # else就是采用统一高斯核
        else:
            # 采用统一的kernel_size=15,sigma=4
            row2 = center_row + int((kernel_size - 1) / 2)
            row1 = center_row - int((kernel_size - 1) / 2)
            col2 = center_col + int((kernel_size - 1) / 2)
            col1 = center_col - int((kernel_size - 1) / 2)

        # 越界检测
        if col1 < 0:
            col1 = 0
        if row1 < 0:
            row1 = 0
        if col2 >= 480:
            col2 = 479
        if row2 >= 300:
            row2 = 299

        # 产生高斯核
        size = (row2 - row1 + 1, col2 - col1 + 1)
        # print("size:",size)
        gaussian_kernel = generate_gaussian_kernel(kernel_size=size, sigma=sigma)
        # print("size of gaussian kernel",np.shape(gaussian_kernel))

        # 打散操作
        density_map[row1:row2 + 1, col1:col2 + 1] += gaussian_kernel

        # 将上面盖住
        density_map[0:35,:]=0

    # 保存density map和它的image
    np.save(density_path + '/' + '{:0>5d}.npy'.format(i + 1), density_map)
    # np.savetxt(density_path+'/'+'{:0>5d}.csv'.format(i+1),density_map,delimiter=',')

    if np.sum(density_map) != 0:
        density_map_image = density_map / np.max(density_map) * 255
        density_map_image.astype('uint8')
    else:
        density_map_image = np.zeros((1200, 1920))
    cv2.imwrite(density_image_path + '/' + 'den_image_{:0>5d}.jpg'.format(i + 1), density_map_image)

    # 打印进度
    if (i+1)%100==0:
        print('density map generate finished {}'.format(i+1))

    print("{} processd".format(gt))
