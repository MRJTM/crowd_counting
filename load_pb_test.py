import tensorflow as tf
import numpy as np
import time
import cv2
from data_process_before_train import learning_rate_type

"""-----------------------------------------------定义识别函数-----------------------------------------"""


def recognize(jpg_path, pb_file_path,gt_path, input_size=(),output_size=()):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        # 打开.pb模型
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print("tensors:", tensors)

        # 在一个session中去run一个前向
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            op = sess.graph.get_operations()

            # 打印图中有的操作
            for i, m in enumerate(op):
                print('op{}:'.format(i), m.values())

            input_x = sess.graph.get_tensor_by_name("input_1:0")  # 具体名称看上一段代码的input.name
            print("input_X:", input_x)

            predict_den = sess.graph.get_tensor_by_name("conv2d_5/Relu:0")  # 具体名称看上一段代码的output.name
            print("Output:", predict_den)

            # 读入图片
            img = cv2.imread(jpg_path, 0)
            img = cv2.resize(img, (input_size[1], input_size[0]))
            # 显示图片
            cv2.imshow("test_image",img)

            # 图片做预处理
            img = img / 255
            img[0:35,:]=0
            img = np.reshape(img, (1, input_size[0], input_size[1], 1))
            print("img data type:", img.dtype)

            t1=time.time()
            out_den = sess.run(predict_den, feed_dict={input_x: img})
            out_den=np.reshape(out_den,output_size)
            t2=time.time()
            print("forward time cost: {}s".format(t2-t1))

            # 打印输出的density map
            cv2.imshow("density",out_den/np.max(out_den))

            # 打印输出人数
            print("predicted num:",np.sum(out_den))

            # 打印实际的人数
            ground_truth=np.load(gt_path)
            print("ground truth:",np.sum(ground_truth))



"""----------------------------------导入pb模型进行测试-----------------------------------"""
model_name = 'hongqiao'
data_type = 5
learning_rate = 0.00001
lr_m, lr_index = learning_rate_type(learning_rate)
batch_size = 1
epoch = 50
input_size = (300, 480)
output_size = (75, 120)

pb_path = 'saved_models/{}_dt{}_lr{}e{}_bs{}_ep{}' \
          '.pb'.format(model_name, data_type, lr_m, lr_index, batch_size, epoch)

img = 'test_images/frame_01120.jpg'
gt_path='test_images/frame_01120.npy'

recognize(img, pb_path,gt_path, input_size,output_size)
cv2.waitKey(0)
