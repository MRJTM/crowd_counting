"""
to generate train dataset and test dataset
"""

import os
import sys
import numpy as np
import numpy.random as random

def divide_into_train_test(dataset_original_path,ratio_of_test):

    # 判断数据集根目录是否存在
    if not os.path.exists(dataset_original_path):
        print("{} does not exist!".format(dataset_original_path))
        exit(0)

    # 创建test的文件夹
    test_image_path = dataset_original_path + '/test_images'
    test_density_map_path = dataset_original_path + '/test_density_map'
    test_ground_truth_path = dataset_original_path + '/test_ground_truth'

    test_paths=[test_image_path,test_density_map_path,test_ground_truth_path]

    for test_path in test_paths:
        if not os.path.exists(test_path):
            os.mkdir(test_path)

    # 按照比例等间隔抽取图片作为测试集
    num_total=len(os.listdir(dataset_original_path+'/image'))
    num_test=int(num_total*ratio_of_test)
    print("num of test:",num_test)

    # 计算采样间隔
    sample_interval=np.floor(num_total/num_test)

    # 产生一个从0～sample_interval的随机数
    random.seed()
    start_index=random.randint(0,sample_interval)
    print("start index:",start_index)

    # 从总的数据集中抽出num_test个样本到对应的文件夹中
    for i in range(num_test):
        print("test dataset generation finished {:.3f}".format(100*i/num_test))
        index=int(start_index+i*sample_interval)
        image_name="{:0>5d}.jpg".format(index)
        density_map_name="{:0>5d}.jpg".format(index)
        ground_truth_name="{:0>5d}.npy".format(index)

        image_source_path=dataset_original_path+'/'+'image'+'/'+image_name
        density_map_source_path=dataset_original_path+'/'+'density_map'+'/'+density_map_name
        ground_truth_source_path=dataset_original_path+'/'+'ground_truth'+'/'+ground_truth_name

        os.system("sudo mv {} {}".format(image_source_path,test_image_path))
        os.system("sudo mv {} {}".format(density_map_source_path, test_density_map_path))
        os.system("sudo mv {} {}".format(ground_truth_source_path, test_ground_truth_path))

    # 最后再把剩余的总文件夹改名为train的文件夹
    original_image_path=dataset_original_path+'/image'
    original_density_map_path=dataset_original_path+'/density_map'
    original_ground_truth_path=dataset_original_path+'/ground_truth'

    train_image_path=dataset_original_path+'/train_images'
    train_density_map_path=dataset_original_path+'/train_density_map'
    train_ground_truth_path=dataset_original_path+'/train_ground_truth'

    os.system("mv {} {}".format(original_image_path, train_image_path))
    os.system("mv {} {}".format(original_density_map_path, train_density_map_path))
    os.system("mv {} {}".format(original_ground_truth_path, train_ground_truth_path))


"""------------------------------------------执行程序---------------------------------------------------"""
dataset_path=sys.argv[1]
ratio_test=float(sys.argv[2])
print("type of ratio_test:",ratio_test)
print("ratio_test:",ratio_test)


divide_into_train_test(dataset_path,ratio_test)
