import cv2
from keras.models import *
from data_process_before_train import learning_rate_type
from data_process_before_train import adjust_brightness_contrast
import numpy as np
import sys
import time
import os

"""强制使用CPU"""
# os.environ["CUDA_VISIBLE_DEVICES"]=""

"""---------------导入模型--------------------"""
model_name = 'hongqiao'
data_type = 5
learning_rate = 0.00001
lr_m, lr_index = learning_rate_type(learning_rate)
batch_size = 1
epoch = 50

new_model=load_model('saved_models/{}_dt{}_lr{}e{}_bs{}_ep{}'
                     '.h5'.format(model_name, data_type, lr_m, lr_index, batch_size, epoch))

"""----------------读入视频-----------------"""
# video_path='../../../video/videos1/vid0.ts'
video_path = sys.argv[1]
brighten = int(sys.argv[2])  # 0表示不提高亮度，1表示提高亮度
cap = cv2.VideoCapture()
cap.open(video_path)

"""---------------对视频帧进行处理-----------------"""
# 设置处理的帧间隔，看多少帧间隔，配合运算时间，可以实现良好的显示效果
frame_interval = 15
# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total frame:", total_frames)

# 对视频进行分析
num_people = 0
time_last = time.time()
for i in range(total_frames):
    _, frame = cap.read()
    # 改变亮度
    if brighten == 1:
        frame=adjust_brightness_contrast(frame,1.5,+50)

    # 每隔frame_interval去刷新一下人数
    if (i + 1) % frame_interval == 0:
        time_start = time.time()
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (480, 300))

        image[0:35, :] = 0
        image = image / 255
        # image = (image - 127.5) / 128
        image_input = np.reshape(image, (1, image.shape[0], image.shape[1], 1))
        density_map = new_model.predict(image_input)
        num_people = np.round(np.sum(density_map))

        # 保存density map
        density_map = np.reshape(density_map, (density_map.shape[1], density_map.shape[2]))
        density_map_image = density_map / np.max(density_map) * 255 * 10
        density_map_image = density_map_image.astype("uint8")
        density_map_image = cv2.resize(density_map_image, (density_map_image.shape[1] * 2,
                                                           density_map_image.shape[0] * 2))
        cv2.imshow('density_map', density_map_image)
        cv2.imwrite('../density_map/den_{:0>5d}.jpg'.format(i), density_map)
        # time_now = time.time()
        # used_time=time_now-time_last
        # time_last=time_now
        time_end = time.time()
        used_time = time_end - time_start
        print("used_time for a update:", used_time)

        cv2.waitKey(19950731)

    # frame[0:180, :] = 0
    frame = cv2.resize(frame, (1400, 900))
    # 把人数写到图片上
    cv2.putText(frame, "num_people:{}".format(num_people),
                (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    cv2.imshow('video', frame)

    if cv2.waitKey(1) == 27:
        break

# 关闭cap
cap.release()
