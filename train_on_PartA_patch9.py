from keras.models import *
from keras.layers import *
import keras
import os
import numpy as np
import numpy.random as random
import cv2
import pandas as pd

# 配置超参数
learning_rate = 0.0001
train_data_path = 'dataset/shanghaitech_part_A_patches_9/'
val_data_path = 'dataset/shanghaitech_part_A_patches_9/'

epoch = 6

# 配置网络结构
inputs = Input(shape = (None, None, 1))
conv_m = Conv2D(20, (7, 7), padding = 'same', activation = 'relu')(inputs)
conv_m = MaxPooling2D(pool_size = (2, 2))(conv_m)
conv_m = (conv_m)
conv_m = Conv2D(40, (5, 5), padding = 'same', activation = 'relu')(conv_m)
conv_m = MaxPooling2D(pool_size = (2, 2))(conv_m)
conv_m = Conv2D(20, (5, 5), padding = 'same', activation = 'relu')(conv_m)
conv_m = Conv2D(10, (5, 5), padding = 'same', activation = 'relu')(conv_m)
result = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_m)

model = Model(inputs = inputs, outputs = result)

def MAE(y_true, y_pred):
    return abs(K.sum(y_true) - K.sum(y_pred))


def MSE(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))

model.compile(loss=MAE, optimizer=keras.optimizers.Adam(lr=learning_rate),metrics=[MAE,MSE])


def generate_function(dataset_path, file_list):
    while 1:
        for K, file in enumerate(file_list):
            image_path = dataset_path + 'train/' + file
            image = cv2.imread(image_path,0)
            image = np.array(image)
            image = (image - 127.5) / 128
            image = np.reshape(image, (1, image.shape[0], image.shape[1], 1))

            gt_path = dataset_path + 'train_den/' + '{}.csv'.format(file.split(".")[0])
            den = pd.read_csv(gt_path, sep=',', header=None).as_matrix()
            den = den.astype(np.float32, copy=False)
            den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
            for i in range(len(den_quarter)):
                for j in range(len(den_quarter[0])):
                    for p in range(4):
                        for q in range(4):
                            den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
            den_quarter = np.reshape(den_quarter, (1, den_quarter.shape[0], den_quarter.shape[1], 1))

            yield (image, den_quarter)


# 装载训练集
file_list = os.listdir(train_data_path + 'train')
random.seed()
random.shuffle(file_list)
num_sample = len(file_list)

model.fit_generator(generate_function(dataset_path=train_data_path, file_list=file_list),
                    steps_per_epoch=num_sample,
                    epochs=epoch,
                    )

# 保存模型
json_string = model.to_json()
open('trained_models/PartA_model1.json', 'w').write(json_string)
model.save_weights('trained_models/PartA_weights1.h5')
