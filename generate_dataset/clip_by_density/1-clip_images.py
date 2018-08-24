# coding=utf-8
import cv2
import scipy.io as io
import numpy as np
import os


def count_std(density_map):
    # 计算1/9大小的patch的尺寸
    patch_rows = int(density_map.shape[0] / 3)
    patch_cols = int(density_map.shape[1] / 3)

    # 获取9个patch的各自的人数，生成一个列表
    patch_count_list = []
    for row in range(3):
        for col in range(3):
            patch_count = np.sum(density_map[patch_rows * row:patch_rows * (row + 1),
                                 patch_cols * col:patch_cols * (col + 1)])
            patch_count_list.append(patch_count)

    # 计算9个patch人数列表的标准差
    std = np.std(patch_count_list)
    return std


def clip_image_with_one_box(image_path, gt_path, save_path, max_people=1000, min_people=100, init_boxsize=(100, 100),
                            box_shrink_rate=0.05, pixel_steps=10, max_std=1000,shape_type=0,dataset_name='UCF_CC_50'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print('Unable to read image {}'.format(image_path))
        exit(0)
    else:
        input_image_name = image_path.split('/')[-1]
        input_image_name = input_image_name.split('.')[0]
        # print(input_image_name)
        # print("size of input image:", np.shape(image))
        # cv2.imshow("input image", image)

    # 载入ground truth的density map
    if dataset_name=='UCF_CC_50':
        gt_mat = np.floor(io.loadmat(gt_path)['annPoints'])
    elif dataset_name=='ShanghaiTech':
        gt_mat = io.loadmat(gt_path)['image_info'][0][0][0][0][0]
    else:
        print('unknown dataset name,you can input dataset_name="UCF_CC_50"or"ShanghaiTech"')
        exit(0)

    # 根据gt_mat，生成对应的density_map的矩阵
    density_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # print("size of density map:", np.shape(density_map))
    for i in range(gt_mat.shape[0]):
        y = int(gt_mat[i, 1])
        x = int(gt_mat[i, 0])
        if y >=image.shape[0]:
            continue
        if x >=image.shape[1]:
            continue
        density_map[y, x] = 1

    # 显示density map
    density_map_image = density_map * 255
    # cv2.imshow("density_map", density_map_image)

    # 计算像素遍历步长
    row_step = int((density_map.shape[0] - 100) / pixel_steps)
    col_step = int((density_map.shape[1] - 100) / pixel_steps)
    # print('row_step:', row_step)
    # print('col_step:', col_step)

    # 计算窗口缩小步长
    box_row_step = int(init_boxsize[0] * box_shrink_rate)
    box_col_step = int(init_boxsize[1] * box_shrink_rate)
    # print('box_row_step:', box_row_step)
    # print('box_col_step:', box_col_step)

    # 以每一个像素为起点，在起始框大小的基础上，不断缩小框
    i = 0
    for row in range(0, image.shape[0] - 99, row_step):
        for col in range(0, image.shape[1] - 99, col_step):
            boxsize = [init_boxsize[0], init_boxsize[1]]
            # 若初始窗口大小超过图像边界，就先缩小窗口
            if row + boxsize[0] > density_map.shape[0] or col + boxsize[1] > density_map.shape[1]:
                rate = np.max([boxsize[0] / (image.shape[0] - row), boxsize[1] / (image.shape[1] - col)])
                boxsize[0] /= rate
                boxsize[1] /= rate
                # 判断如果缩小后行数或者列数<100，就放弃这个像素点
                if np.min([boxsize[0], boxsize[1]]) < 100:
                    continue
            # 若当前窗口大小在图像范围内，就不断缩小窗口并判断人数,但是窗口不能小于100x100的大小
            else:
                while boxsize[0] > 100 and boxsize[1] > 100:
                    density_map_clip = density_map[row:row + boxsize[0], col:col + boxsize[1]]
                    num = np.sum(density_map_clip)
                    # 若窗口人数达不到要求，就直接结束对这个像素点的遍历
                    if num < min_people or num > max_people:
                        break
                    # 若窗口人数达到了，但是达不到方差要求，就缩小窗口
                    elif count_std(density_map_clip) > max_std:
                        boxsize[0] -= box_row_step
                        boxsize[1] -= box_col_step
                    # 若两者都满足要求了，就保存，并对下一个像素点进行遍历
                    else:
                        image_clip = image[row:row + boxsize[0], col:col + boxsize[1]]
                        density_image_clip = density_map_image[row:row + boxsize[0], col:col + boxsize[1]]

                        image_clip_name = save_path + '{}_shape{}_clip_{:0>4d}.jpg'.format(input_image_name,shape_type, i)
                        density_image_clip_name = save_path + '{}_shape{}_density_clip_{:0>4d}.jpg'.format(input_image_name,shape_type, i)
                        density_clip_name = save_path + '{}_shape{}_density_clip_{:0>4d}.npy'.format(input_image_name,shape_type, i)

                        cv2.imwrite(image_clip_name, image_clip)
                        cv2.imwrite(density_image_clip_name, density_image_clip)
                        np.save(density_clip_name, density_map_clip)
                        i += 1
                        break


"""-----------------------------------------------执行程序-----------------------------------------------------"""
"""
不同数据集只需改一下dataset name即可
对与ShanghaiTech的数据集还要配置好part_name和train_or_test
"""
# dataset_name='ShanghaiTech'
part_name='B'
train_or_test='test'

dataset_name='UCF_CC_50'

if dataset_name=='UCF_CC_50':
    image_path = '../UCF_CC_50/'
    gt_path = '../UCF_CC_50/'
    XL_save_path = '/home/czj/UCF_CC_50_Clip/XL_density/'
    L_save_path = '/home/czj/UCF_CC_50_Clip/L_density/'
    M_save_path = '/home/czj/UCF_CC_50_Clip/M_density/'
    S_save_path = '/home/czj/UCF_CC_50_Clip/S_density/'
    num_pictures=50
else:
    image_path = '../ShanghaiTech_Dataset/part_{}_final/{}_data/images/'.format(part_name,train_or_test)
    gt_path = '../ShanghaiTech_Dataset/part_{}_final/{}_data/ground_truth/'.format(part_name,train_or_test)
    XL_save_path = '/home/czj/ShanghaiTech/P{}_{}/XL_density/'.format(part_name,train_or_test)
    L_save_path = '/home/czj/ShanghaiTech/P{}_{}/L_density/'.format(part_name,train_or_test)
    M_save_path = '/home/czj/ShanghaiTech/P{}_{}/M_density/'.format(part_name,train_or_test)
    S_save_path = '/home/czj/ShanghaiTech/P{}_{}/S_density/'.format(part_name,train_or_test)
    num_pictures_list = {'PA_train': 300, 'PA_test': 182, 'PB_train': 400, 'PB_test': 316}
    num_pictures = num_pictures_list['P{}_{}'.format(part_name, train_or_test)]


# 全局的配置
save_paths = [XL_save_path, L_save_path, M_save_path, S_save_path]
num_people = [[10000, 120],
              [120, 60],
              [60, 20],
              [20, 0]]

max_std_list = [12, 9, 6, 15]

# 5种不同的box_size
init_boxsize_list=[(500,600),
                   (480,600),
                   (450,600),
                   (400,600),
                   (300,600)]


# 对每张图片和对应的ground truth进行处理
for i in range(1, num_pictures+1):
    if dataset_name=='UCF_CC_50':
        image_name = image_path + '{}.jpg'.format(i)
        gt_name = gt_path + '{}_ann.mat'.format(i)
        print('{}.jpg is being clipped from UCF_CC_50....'.format(i))
    elif dataset_name=='ShanghaiTech':
        image_name = image_path + 'IMG_{}.jpg'.format(i)
        gt_name = gt_path + 'GT_IMG_{}.mat'.format(i)
        print('IMG_{}.jpg is being clipped from ShanghaiTech......'.format(i))


    # 不同人数密度的循环
    for k, save_path in enumerate(save_paths):
        max_people = num_people[k][0]
        min_people = num_people[k][1]
        max_std = max_std_list[k]
        print('\t\t num_people:[{}-{}],max_std:{}'.format(min_people,max_people,max_std))

        # 针对不同比例尺寸都去切一下
        for shape_type,init_boxsize in enumerate(init_boxsize_list):
            # print('\t\t\t\t init_boxsize:{}'.format(init_boxsize))
            clip_image_with_one_box(image_name,                 # 图片路径
                                    gt_name,                    # ground truth的路径
                                    save_path,                  # 保存截取结果的路径
                                    pixel_steps=10,             # 像素遍历步长
                                    init_boxsize=init_boxsize,  # 起始框大小
                                    box_shrink_rate=0.1,        # 窗口缩小步长
                                    max_people=max_people,      # 切片中的最大人数要求
                                    min_people=min_people,      # 切片中的最小人数要求
                                    max_std=max_std,            # 最大标准差约束
                                    shape_type=shape_type,      # boxsize的尺寸类型
                                    dataset_name=dataset_name   # 数据集名称
                                    )

# 对不同密度的图片的std测试
# 最大密度>120人的情况的测试
# for i in range(1, 51):
#     image_name = image_path + '{}.jpg'.format(i)
#     gt_name = gt_path + '{}_ann.mat'.format(i)
#
#     max_people = 20
#     min_people = 0
#     max_std =15
#     clip_image_with_one_box(image_name,                 # 图片路径
#                             gt_name,                    # ground truth的路径
#                             S_save_path,               # 保存截取结果的路径
#                             pixel_steps=15,             # 像素遍历步长
#                             init_boxsize=(300, 450),    # 起始框大小
#                             box_shrink_rate=0.1,        # 窗口缩小步长
#                             max_people=max_people,      # 切片中的最大人数要求
#                             min_people=min_people,      # 切片中的最小人数要求
#                             max_std=max_std)            # 最大标准差约束


cv2.waitKey(0)
