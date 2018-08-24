import numpy as np
import cv2
import os
import numpy.random as random
from multiprocessing import Process, Manager


def learning_rate_type(lr):
    i=0
    while lr<1:
        lr*=10
        i+=1
    mag=int(lr)
    return mag,i


def generate_gaussian_kernel(kernel_size=(15, 15), sigma=3):
    rows=kernel_size[0]
    cols=kernel_size[1]
    if rows%2==1:
        m1=int((rows-1)/2)
        m2=int((rows-1)/2)+1
    else:
        m1=int(rows/2)
        m2=int(rows/2)

    if cols%2==1:
        n1 = int((cols - 1) / 2)
        n2 = int((cols - 1) / 2) + 1
    else:
        n1=int(cols/2)
        n2=int(cols/2)

    y, x = np.ogrid[-m1:m2, -n1:n2]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gaussian_disperse(density_map, kernel_size=15, sigma=3):
    max_rows = density_map.shape[0]
    max_cols = density_map.shape[1]
    density_map_blur = np.zeros((max_rows, max_cols))

    # 根据density_map中的点的位置，放上高斯核
    rows = np.where(density_map == 1)[0]
    cols = np.where(density_map == 1)[1]
    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]
        # print('position:({},{})'.format(row,col))
        row1 = row - int(np.floor(kernel_size / 2))
        row2 = row + int(np.floor(kernel_size / 2))
        col1 = col - int(np.floor(kernel_size / 2))
        col2 = col + int(np.floor(kernel_size / 2))
        # print("r1:{},r2:{},col1:{},col2:{}".format(row1,row2,col1,col2))
        change_kernel = False
        if row1 < 0:
            row1 = 0
            change_kernel = True
        if row2 > max_rows - 1:
            row2 = max_rows - 1
            change_kernel = True
        if col1 < 0:
            col1 = 0
            change_kernel = True
        if col2 > max_cols - 1:
            col2 = max_cols - 1
            change_kernel = True

        if change_kernel == False:
            size = (kernel_size, kernel_size)
        else:
            size = (row2 - row1 + 1, col2 - col1 + 1)
        gaussian_kernel = generate_gaussian_kernel(kernel_size=size, sigma=sigma)
        # print('generated kernel size:',np.shape(gaussian_kernel))
        # print('density_map_patch size:',np.shape(density_map_blur[row1:row2+1,col1:col2+1]))
        density_map_blur[row1:row2 + 1, col1:col2 + 1] += gaussian_kernel

    return density_map_blur


def resize_density_map(density_map, new_size=(300, 300)):
    row_rate = density_map.shape[0] / new_size[0]
    col_rate = density_map.shape[1] / new_size[1]
    new_density_map = np.zeros(new_size)
    rows = np.where(density_map == 1)[0]
    cols = np.where(density_map == 1)[1]
    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]
        new_row = int(row / row_rate)
        new_col = int(col / col_rate)
        new_density_map[new_row, new_col] = 1
    return new_density_map


def load_whole_dataset(dataset_path, input_size=(300, 300), output_size=(60, 80),
                       dp_enlarge_rate=40, gaussian_kernel_size=23, gaussian_sigma=4,
                       load_num=10000):
    train_X = []
    train_gt = []
    for i in range(load_num):
        train_X.append(1)
        train_gt.append(1)

    random.seed()
    file_list = os.listdir(dataset_path + 'train_images')
    random.shuffle(file_list)
    for k, image in enumerate(file_list):
        total_num = load_num
        if k % int(total_num / 50) == 0:
            print("data loading finished {}%".format(100 * k / total_num))
        img = cv2.imread(dataset_path + 'train_images/' + image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resize = cv2.resize(img, (input_size[1], input_size[0]))
        train_X[k] = img_resize

        gt_name = "{}.npy".format(image.split(".")[0])
        gt_data = np.load(dataset_path + 'train_ground_truth/' + gt_name)
        gt_resize = resize_density_map(gt_data, output_size)
        gt_resize = gaussian_disperse(gt_resize, kernel_size=gaussian_kernel_size,
                                      sigma=gaussian_sigma) * dp_enlarge_rate
        train_gt[k] = gt_resize

        if k >= load_num - 1:
            break

    train_X = np.reshape(train_X, (len(train_X), input_size[0], input_size[1], 1))
    train_gt = np.reshape(train_gt, (len(train_gt), output_size[0], output_size[1], 1))
    print("size of train_X:", np.shape(train_X))
    print("size of train_gt:", np.shape(train_gt))

    return train_X, train_gt


def generate_val(dataset_path, input_size=(300, 300), output_size=(60, 80), val_num=100):
    val_X = [x for x in range(val_num)]
    val_gt = [y for y in range(val_num)]
    name_list = os.listdir(dataset_path + 'test_images')
    total_num = int(len(name_list))
    num_interval = int(np.floor(total_num / val_num))
    random.seed()
    start_index = random.randint(0, num_interval)
    i=0
    for k, image in enumerate(name_list):
        if (k - start_index) % num_interval == 0 and k / num_interval < val_num:
            img = cv2.imread(dataset_path + 'test_images/' + image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_resize = cv2.resize(img, (input_size[1], input_size[0]))
            val_X[i]=img_resize

            gt_name = "{}.npy".format(image.split(".")[0])
            gt_data = np.load(dataset_path + 'test_ground_truth/' + gt_name)
            gt_resize = resize_density_map(gt_data, output_size)
            gt_resize = gaussian_disperse(gt_resize)
            val_gt[i]=gt_resize
            i+=1
        if k / num_interval > val_num:
            break

    val_X = np.reshape(val_X, (len(val_X), input_size[0], input_size[1], 1))
    val_gt = np.reshape(val_gt, (len(val_gt), output_size[0], output_size[1], 1))
    print("size of val_X:", np.shape(val_X))
    print("size of val_gt:", np.shape(val_gt))

    return (val_X, val_gt)


def generate_function(dataset_path, file_list,
                      gaussian_kernel_size=15, gaussian_sigma=4):
    while 1:
        random.seed()
        random.shuffle(file_list)
        for K, file in enumerate(file_list):
            image_path = dataset_path + 'train_images/' + file
            image = cv2.imread(image_path,0)
            X = np.array(image)
            X = X/255
            X = np.reshape(X,(1,X.shape[0],X.shape[1],1))

            gt_name = "{}.npy".format(file.split(".")[0])
            gt_data = np.load(dataset_path + 'train_ground_truth/' + gt_name)
            den = gaussian_disperse(gt_data,
                                   kernel_size=gaussian_kernel_size,
                                   sigma=gaussian_sigma)

            den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
            for i in range(len(den_quarter)):
                for j in range(len(den_quarter[0])):
                    for p in range(4):
                        for q in range(4):
                            den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
            den_quarter = np.reshape(den_quarter, (1, den_quarter.shape[0], den_quarter.shape[1], 1))

            yield (X, den_quarter)

# 装载数据的函数
def load_data(images_path, gt_path, data_type=1, train_or_val='train', input_size=(), output_size=()):
    # 装载训练集
    image_list = os.listdir(images_path)
    random.seed()
    random.shuffle(image_list)
    num_sample = len(image_list)

    X = [x for x in range(num_sample)]
    gt = [y for y in range(num_sample)]

    for k, file in enumerate(image_list):
        if (k + 1) % 50 == 0:
            print('{} data load finished {}/{}'.format(train_or_val, k + 1, num_sample))
        image_path = images_path + '/' + file
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (input_size[1], input_size[0]))
        image = np.array(image)

        if (data_type == 2) or (data_type==6):
            image = (image - 127.5) / 128
        elif data_type != 3:
            image = image / 255

        # 上边用一个mask遮住
        image[0:35, :] = 0
        X[k] = image

        den_path = gt_path + '/' + '{}.npy'.format(file.split(".")[0])
        den = np.load(den_path)
        den_quarter = np.zeros((output_size[0], output_size[1]))
        rows_rate = int(input_size[0] / output_size[0])
        cols_rate = int(input_size[1] / output_size[1])
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(rows_rate):
                    for q in range(cols_rate):
                        den_quarter[i][j] += den[i * rows_rate + p][j * cols_rate + q]
        gt[k] = den_quarter

    X = np.reshape(X, (len(image_list), input_size[0], input_size[1], 1))
    gt = np.reshape(gt, (len(image_list), output_size[0], output_size[1], 1))
    print("size of train_X:", np.shape(X))
    print("size of train_gt:", np.shape(gt))

    return X, gt

# 图像亮度调节函数
def adjust_brightness_contrast(src1,k,b):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道

    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, k, src2, 1 - k, b)
    return dst



