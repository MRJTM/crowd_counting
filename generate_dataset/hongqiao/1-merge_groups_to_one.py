"""
this script is to merge three image folders and gt folders int one fold pair
and rename them
"""

import os
import sys

"""---------------------配置路径------------------------"""
data_path=sys.argv[1]
folder_numbers=int(sys.argv[2])
# data_path='../../../dataset/hongqiao/'
images_path=data_path+'images/'
gts_path=data_path+'gt/'

image_out_path=images_path+'images'
gt_out_path=gts_path+'gt'

# 若输出路径不存在，就创建相关路径
if not os.path.exists(image_out_path):
    # os.system("sudo mkdir {}".format(image_out_path))
    os.mkdir(image_out_path)
if not os.path.exists(gt_out_path):
    # os.system("sudo mkdir {}".format(gt_out_path))
    os.mkdir(gt_out_path)

"""--------------------复制图片和gt，并重命名-------------------------"""
index=1
for i in range(1,folder_numbers+1):
    image_path=images_path+'images{}'.format(i)
    gt_path=gts_path+'gt{}'.format(i)
    images_list=os.listdir(image_path)
    gt_list = os.listdir(gt_path)

    images_list.sort()
    gt_list.sort()

    num_images=len(images_list)
    num_gt=len(gt_list)

    for image,gt in zip(images_list,gt_list):
        # print("image,gt:",(image,gt))
        image_source_name=image_path+'/'+image
        gt_source_name=gt_path+'/'+gt

        image_target_name=image_out_path+'/'+'{:0>5d}.jpg'.format(index)
        gt_target_name=gt_out_path+'/'+'{:0>5d}.xml'.format(index)

        if index%50==0:
            print("merge finished {}".format(index))

        os.system("sudo cp {} {}".format(image_source_name,image_target_name))
        os.system("sudo cp {} {}".format(gt_source_name,gt_target_name))

        index += 1