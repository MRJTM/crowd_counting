"""
本脚本是为了改变图像的亮度来实现数据增强的
"""

import cv2
import numpy.random as random
import os
from data_process_before_train import adjust_brightness_contrast

"""------------------------配置相关路径-----------------------------"""
image_path = '../../../dataset/hongqiao/images/images'
density_map_path = '../../../dataset/hongqiao/gt/density_map'
density_image_path = '../../../dataset/hongqiao/gt/density_image'

save_path_for_planA = '../../../dataset/hongqiao/images/images_aug_A'
save_path_for_planB = '../../../dataset/hongqiao/images/images_aug_B'
gt_path = '../../../dataset/hongqiao/gt'

# 配置数据增强方案
# 1表示直接在原图片上改，不另存为
# 2表示另存为新的图片
augmentation_plan = 1

"""---------------------读取文件改变亮度并另存为------------------------"""
image_list = os.listdir(image_path)
density_map_list = os.listdir(density_map_path)
density_image_list = os.listdir(density_image_path)

image_list.sort()
density_map_list.sort()
density_image_list.sort()

# 获取文件总数
total_num = len(image_list)
print("total images to process:", total_num)

# 对每张图片进行处理
for i in range(total_num):
    # 打印进度
    if (i + 1) % 50 == 0:
        print("{}/{} images have processed".format(i + 1, total_num))
    # 获取文件名
    image = image_list[i]
    density_map = density_map_list[i]
    density_image = density_image_list[i]
    # 读入图片
    image_name = image_path + '/' + image
    img = cv2.imread(image_name)
    # 对每张图片进行处理
    if augmentation_plan == 1:
        # 产生随机数k和b
        random.seed()
        k = random.uniform(0.8, 1.5)
        b = random.randint(-50, 50)
        # 改变图片的亮度
        img_aug = adjust_brightness_contrast(img, k, b)
        # 将图片保存到另一个文件夹中
        save_name = save_path_for_planA + '/' + image
        cv2.imwrite(save_name, img_aug)
    elif augmentation_plan == 2:
        random.seed()
        k = random.uniform(0.8, 1.5)
        # 产生一张变亮的图片
        b = random.randint(20, 50)
        img_light = adjust_brightness_contrast(img, k, b)
        # 产生一张变暗的图片
        b = random.randint(-50, -20)
        img_dark = adjust_brightness_contrast(img, k, b)
        # 保存图片以及对应的density_map和density_image
        img_origin_path = save_path_for_planB + '/' + image
        img_light_path = save_path_for_planB + '/' + '{:0>5d}.jpg'.format(i + 1 + total_num)
        img_dark_path = save_path_for_planB + '/' + '{:0>5d}.jpg'.format(i + 1 + 2 * total_num)
        os.system("cp {} {}".format(image_name, img_origin_path))
        cv2.imwrite(img_light_path, img_light)
        cv2.imwrite(img_dark_path, img_dark)

        density_map_source_path = density_map_path + '/' + density_map
        density_map_origin_path = gt_path + '/' + 'density_map_aug' + '/' + '{:0>5d}.npy'.format(i + 1)
        density_map_light_path = gt_path + '/' + 'density_map_aug' + '/' + '{:0>5d}.npy'.format(i + 1 + total_num)
        density_map_dark_path = gt_path + '/' + 'density_map_aug' + '/' + '{:0>5d}.npy'.format(i + 1 + total_num * 2)
        os.system("cp {} {}".format(density_map_source_path, density_map_origin_path))
        os.system("cp {} {}".format(density_map_source_path, density_map_light_path))
        os.system("cp {} {}".format(density_map_source_path, density_map_dark_path))

        density_image_source_path = density_image_path + '/' + density_image
        density_image_origin_path = gt_path + '/' + 'density_image_aug' + '/' + 'den_image_{:0>5d}.jpg'.format(i + 1)
        density_image_light_path = gt_path + '/' + 'density_image_aug' + '/' + 'den_image_{:0>5d}.jpg'.format(
            i + 1 + total_num)
        density_image_dark_path = gt_path + '/' + 'density_image_aug' + '/' + 'den_image_{:0>5d}.jpg'.format(
            i + 1 + total_num * 2)
        os.system("cp {} {}".format(density_image_source_path, density_image_origin_path))
        os.system("cp {} {}".format(density_image_source_path, density_image_light_path))
        os.system("cp {} {}".format(density_image_source_path, density_image_dark_path))
