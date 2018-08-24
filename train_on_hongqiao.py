from models import single_colum_CNN
import keras
import numpy as np
import keras.backend as K
from data_process_before_train import learning_rate_type
from data_process_before_train import load_data
from keras.models import load_model,save_model
import os
from tools.trainingmonitor import TrainingMonitor

"""--------------------------配置网络超参数-----------------------------"""
train_data_path = '../dataset/hongqiao/images/images_aug_A'
train_den_path = '../dataset/hongqiao/gt/density_map'

output_path='output'


input_size = (300, 480)
output_size = (75, 120)

learning_rate = 0.00001
lr_m, lr_index = learning_rate_type(learning_rate)
batch_size = 1
epoch = 20


# data_type=1,则image=image/255,=2则image=(image-127.5)/128,=3则保留原像素,均采用固定高斯核
# =4则采用有上限21的变高斯核，=5则采用无上限高斯核，4和5对image的处理同1，
# =6则采用无上限的可变高斯核，image的处理同2
data_type = 5

# 读入数据集的方式,1代表从硬盘读并保存为矩阵，2代表加载保存在矩阵的数据集,3代表载入模型继续训练
data_load_plan = 3

# 数据集名称
dataset_name = 'hongqiao_augA'

"""---------------------------配置网络结构-------------------------------"""
# 若data_load_plan!=1，在需要加载模型
if data_load_plan!=1:
    if data_load_plan==2:
        # 装载网络结构
        model = single_colum_CNN(input_shape=(None, None, 1))
    elif data_load_plan==3:
        # 载入训练好的模型
        model_name = 'hongqiao'
        load_data_type = 4
        load_learning_rate = 0.00001
        load_lr_m, load_lr_index = learning_rate_type(load_learning_rate)
        load_batch_size = 1
        load_epoch = 40

        model=load_model('saved_models/{}_dt{}_lr{}e{}_bs{}_ep{}.h5'.format(
            model_name, load_data_type, load_lr_m, load_lr_index,load_batch_size, load_epoch))

    # 配置loss func和优化方法
    def MAE(y_true, y_pred):
        return abs(K.sum(y_true) - K.sum(y_pred))


    def MSE(y_true, y_pred):
        return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))

    # decay=learning_rate/epoch
    decay=0
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=True)
    # opt=keras.optimizers.Adam(learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=[MAE, MSE])

"""----------------------------训练网络-------------------------------"""


if data_load_plan == 1:
    # 从硬盘读入数据集并保存为矩阵
    train_X, train_gt = load_data(train_data_path, train_den_path,
                                  data_type=data_type, train_or_val='train',
                                  input_size=input_size, output_size=output_size)
    np.save('saved_files/{}_train_X{}.npy'.format(dataset_name, data_type), train_X)
    np.save('saved_files/{}_train_gt{}.npy'.format(dataset_name, data_type), train_gt)
    print("saved data.........")

else:
    # 读入数据
    print("loading data..........")
    train_X = np.load('saved_files/{}_train_X{}.npy'.format(dataset_name, data_type))
    train_gt = np.load('saved_files/{}_train_gt{}.npy'.format(dataset_name, data_type))
    print("loaded data...........")

    # 打印数据集尺寸
    print("[INFO]size of train_X:", np.shape(train_X))
    print("[INFO]size of train_gt:", np.shape(train_gt))

    """------------------------------模型训练和保存----------------------------------"""
    # 训练网络
    figPath = os.path.sep.join([output_path, "{}.png".format(
        os.getpid())])
    jsonPath = os.path.sep.join([output_path, "{}.json".format(
        os.getpid())])
    callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]
    model.fit(train_X, train_gt, batch_size=batch_size, epochs=epoch, shuffle=True,
              validation_split=0.1, callbacks=callbacks, verbose=1)

    # 保存模型
    if data_load_plan==2:
        save_model(model, 'saved_models/{}_dt{}_lr{}e{}_bs{}_ep{}.h5'.format(
            dataset_name, data_type, lr_m, lr_index, batch_size, epoch))
    elif data_load_plan==3:
        save_model(model, 'saved_models/{}_dt{}_lr{}e{}_bs{}_ep{}.h5'.format(
            model_name, data_type, lr_m, lr_index, batch_size, load_epoch + epoch))

