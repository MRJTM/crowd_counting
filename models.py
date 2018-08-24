from keras.models import *
from keras.layers import *

def single_colum_CNN(input_shape=()):
    # 配置网络结构
    inputs = Input(shape=input_shape)
    conv_m = Conv2D(20, (7, 7), padding='same', activation='relu')(inputs)
    conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
    conv_m = (conv_m)
    conv_m = Conv2D(40, (5, 5), padding='same', activation='relu')(conv_m)
    conv_m = MaxPooling2D(pool_size=(2, 2))(conv_m)
    conv_m = Conv2D(20, (5, 5), padding='same', activation='relu')(conv_m)
    conv_m = Conv2D(10, (5, 5), padding='same', activation='relu')(conv_m)
    result = Conv2D(1, (1, 1), padding='same', activation='relu')(conv_m)

    model = Model(inputs=inputs, outputs=result)

    return model