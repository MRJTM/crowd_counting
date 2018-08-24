from keras.models import *
from keras.layers import *
import keras
import os
import numpy as np
import numpy.random as random
import cv2
import pandas as pd
import keras.backend as K
import math
from data_process_before_train import learning_rate_type

# 装载数据的函数
def load_data(data_path, gt_path, data_type='train',input_size=(), output_size=()):
    # 装载训练集
    file_list = os.listdir(data_path)
    random.seed()
    random.shuffle(file_list)
    num_sample = len(file_list)

    X = [x for x in range(num_sample)]
    gt = [y for y in range(num_sample)]

    for k, file in enumerate(file_list):
        if (k + 1) % 50 == 0:
            print('{} data load finished {}/{}'.format(data_type,k + 1, num_sample))
        image_path = data_path + file
        image = cv2.imread(image_path, 0)
        image = np.array(image)
        # image = (image-127.5)/128
        image=image/255
        X[k] = image

        den_path = gt_path + '{}.csv'.format(file.split(".")[0])
        den = pd.read_csv(den_path, sep=',', header=None).as_matrix()
        den = den.astype(np.float32, copy=False)
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        gt[k] = den_quarter

    X = np.reshape(X, (len(file_list), input_size[0], input_size[1], 1))
    gt = np.reshape(gt, (len(file_list), output_size[0], output_size[1], 1))
    print("size of train_X:", np.shape(X))
    print("size of train_gt:", np.shape(gt))

    return X, gt


# 配置超参数
learning_rate = 0.0001
lr_m,lr_index=learning_rate_type(learning_rate)
print("lr_m:",lr_m)
print("lr_index:",lr_index)

train_data_path = 'dataset/shanghaitech_part_B_patches_9/train/'
train_den_path = 'dataset/shanghaitech_part_B_patches_9/train_den/'
val_data_path = 'dataset/shanghaitech_part_B_patches_9/val/'
val_den_path = 'dataset/shanghaitech_part_B_patches_9/val_den/'

input_size = (192, 256)
output_size = (48, 64)

batch_size = 32
epoch = 20

data_type=2

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

model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate),metrics=[MAE,MSE])

# 装载数据集

# train_X, train_gt = load_data(train_data_path, train_den_path,'train',input_size, output_size)
# val_X, val_gt = load_data(val_data_path, val_den_path,'val', input_size, output_size)
#
# np.save('saved_files/PartB_train_X1.npy',train_X)
# np.save('saved_files/PartB_train_gt1.npy',train_gt)
# np.save('saved_files/PartB_val_X1.npy',val_X)
# np.save('saved_files/PartB_val_gt1.npy',val_gt)
# print("saved data.........")

print("loading data..........")
train_X=np.load('saved_files/PartB_train_X{}.npy'.format(data_type))
train_gt=np.load('saved_files/PartB_train_gt{}.npy'.format(data_type))
val_X=np.load('saved_files/PartB_val_X{}.npy'.format(data_type))
val_gt=np.load('saved_files/PartB_val_gt{}.npy'.format(data_type))
print("loaded data...........")

print("size of train_X:", np.shape(train_X))
print("size of train_gt:", np.shape(train_gt))

# 训练网络
model.fit(train_X, train_gt, batch_size=batch_size, epochs=epoch, shuffle=True,
          validation_data=(val_X,val_gt))
score=model.evaluate(val_X,val_gt,batch_size=1)
mae=score[1]
mse=math.sqrt(score[2])
print("mae:",mae)
print("mse:",mse)

# 保存模型
json_string = model.to_json()
open('trained_models/PartB_model_dt{}_lr{}e{}_bs{}_ep{}.json'.format(data_type,lr_m,lr_index,batch_size,epoch), 'w').write(json_string)
model.save_weights('trained_models/PartB_weights_dt{}_lr{}e{}_bs{}_ep{}.h5'.format(data_type,lr_m,lr_index,batch_size,epoch))
