from keras.models import load_model
from data_process_before_train import *
import pandas as pd
from keras.models import model_from_json
import keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""--------------------------测试方案配置-----------------------------------"""
# test_plan=1表示用一张自己找的图片，2表示用自己切的图片，3表示用人家切的ShanghaiTech的图片,
# 4用10张图片计算一个MAE,5用11张虹桥数据集图片进行测试，并计算MAE
test_plan = 1
# 自己找的图片
if test_plan == 1:
    image_number = 631
    # image_path = 'test_images/{:0>5d}.jpg'.format(image_number)
    image_path = 'test_images/IMG_1.jpg'

# 自己切的图片
elif test_plan == 2:
    # image_number = 2437
    # image_number=2247
    # density_level = "S"
    # image_path = '../dataset/final/{}_density/test_images/{:0>5d}.jpg'.format(density_level, image_number)
    # gt_path = '../dataset/final/{}_density/test_ground_truth/{:0>5d}.npy'.format(density_level, image_number)
    image_path = 'test_images/frame_01120.jpg'
    gt_path = 'test_images/frame_01120.npy'

# 人家切的ShanghaiTech的图片
elif test_plan == 3:
    image_number = '2_2'
    image_path = 'dataset/shanghaitech_part_B_patches_9/train/{}.jpg'.format(image_number)
    gt_path = 'dataset/shanghaitech_part_B_patches_9/train_den/{}.csv'.format(image_number)

elif test_plan == 4:
    image_path = 'test_images/crowd'
    gt_list = [23, 38, 12, 11, 82, 23, 49, 33, 800, 353, 0]

elif test_plan == 5:
    image_path = 'test_images/hongqiao'

"""-------------------------载入已经训练好的模型-----------------------------------------"""
# model_name = 'hongqiao'
model_name = 'PartB'
data_type = 1
learning_rate = 0.0001
lr_m, lr_index = learning_rate_type(learning_rate)
batch_size = 1
epoch = 40

new_model=load_model('saved_models/{}_dt{}_lr{}e{}_bs{}_ep{}'
                     '.h5'.format(model_name, data_type, lr_m, lr_index, batch_size, epoch))

"""-----------------------------载入测试图片--------------------------------------------"""
if test_plan < 4:
    image = cv2.imread(image_path, 0)
    rate = image.shape[1] / 256
    image = cv2.resize(image, (int(image.shape[1] / rate), int(image.shape[0] / rate)))
    # image[0:45,:]=0
    cv2.imshow('test image', image)
    image = np.array(image)
    if (data_type == 2) or (data_type == 6):
        image = (image - 127.5) / 128
    elif data_type != 3:
        image = image / 255

"""---------------------------使用模型进行预测-------------------------------------------"""
if test_plan < 4:
    image_input = np.reshape(image, (1, image.shape[0], image.shape[1], 1))

    density_map = new_model.predict(image_input)
    print("predicted num of people:", np.sum(density_map))

    # 打印预测出来的density map
    density_map = np.reshape(density_map, (density_map.shape[1], density_map.shape[2]))

    # 保存输出的density map的矩阵数据
    # np.savetxt('saved_files/predict_dp.csv',density_map,delimiter=',')

    # 将density map放大显示
    density_map=cv2.resize(density_map,(density_map.shape[1]*4,density_map.shape[0]*4))

    # 将density map高斯模糊
    density_map=cv2.GaussianBlur(density_map,(15,15),sigmaX=4,sigmaY=4)

    # 将density map转化为热力图
    density_map=density_map/np.max(density_map)*255
    density_map=density_map.astype(np.uint8)
    density_map=cv2.applyColorMap(density_map,cv2.COLORMAP_JET)

    cv2.imshow("predicted density map", density_map)
elif test_plan == 4:
    MAE = 0
    print("image_name   ground_truth   predicted_num")
    for i in range(1, 12):
        image = cv2.imread(image_path + "crowd{}.jpeg".format(i), 0)
        image = np.array(image)
        if data_type == 1:
            image = image / 255
        elif data_type == 2:
            image = (image - 127.5) / 128

        image_input = np.reshape(image, (1, image.shape[0], image.shape[1], 1))

        density_map = new_model.predict(image_input)
        print("crowd{}.jpeg     {}            {:.3f}".format(i, gt_list[i - 1], np.sum(density_map)))
        MAE += abs(gt_list[i - 1] - np.sum(density_map))
    MAE /= 11
    print("total MAE=", MAE)
elif test_plan == 5:
    MAE = 0
    print("image_name   ground_truth   predicted_num")
    image_list = os.listdir(image_path)
    image_list.sort()
    for image_name in image_list:
        image = cv2.imread(image_path + '/' + image_name, 0)
        image = cv2.resize(image, (480, 300))
        if (data_type == 2) or (data_type == 6):
            image = (image - 127.5) / 128
        elif data_type != 3:
            image = image / 255
        image[0:45, :] = 0
        image_input = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
        density_map = new_model.predict(image_input)

        gt_path = '../dataset/hongqiao/gt/density_map/{}.npy'.format(image_name.split('.')[0])
        gt = np.load(gt_path)
        gt_num = np.sum(gt)
        print("\t{}     {:.3f}            {:.3f}".format(image_name, gt_num, np.sum(density_map)))
        MAE += abs(gt_num - np.sum(density_map))

        # 保存每张图片的predict density map
        density_map = np.reshape(density_map, (75, 120))
        density_map_image = density_map / np.max(density_map) * 255
        density_map_image.astype('uint8')
        cv2.imwrite(
            'saved_files/predict_density_map/predict_den{}_dt{}_lr{}e{}_bs{}_ep{}.jpg'.format(image_name.split('.')[0],
                                                                                              data_type, lr_m, lr_index,
                                                                                              batch_size, epoch),
            density_map_image)
    MAE /= 11
    print("total MAE=", MAE)

"""--------------------------------导入ground truth----------------------------------"""
if test_plan > 1 and test_plan < 4:
    if test_plan == 2:
        gt_data = np.load(gt_path)
        # gt_data = gaussian_disperse(gt_data, kernel_size=15, sigma=4)

    elif test_plan == 3:
        gt_data = pd.read_csv(gt_path, sep=',', header=None).as_matrix()
        gt_data = gt_data.astype(np.float32, copy=False)

    cv2.imshow('ground_truth', gt_data / np.max(gt_data))
    print('ground_truth:', np.sum(gt_data))

    # 打印ground truth
    den_quarter = np.zeros((int(gt_data.shape[0] / 4), int(gt_data.shape[1] / 4)))
    for i in range(len(den_quarter)):
        for j in range(len(den_quarter[0])):
            for p in range(4):
                for q in range(4):
                    den_quarter[i][j] += gt_data[i * 4 + p][j * 4 + q]

    print('ground_truth_resize:', np.sum(den_quarter))

    density_map = density_map / np.max(density_map)
    den_quarter = den_quarter / np.max(den_quarter)
    # result_img = np.hstack((den_quarter, density_map))
    # cv2.imshow('predict vs gt', result_img)

cv2.waitKey(0)
