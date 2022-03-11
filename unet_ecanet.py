from tensorflow.keras.layers import Input, concatenate, Dense, Flatten, BatchNormalization, Dropout, ReLU, Conv2D, Reshape, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Reshape, MaxPooling2D, Conv2DTranspose, Add, GlobalAveragePooling2D, Conv1D, Activation, multiply
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import math
import tensorflow as tf
from tensorflow.keras.regularizers import l2

input_shape = (96,96,1)
Nkeypoints = 15
def eca_layer(inputs_tensor=None,num=None,gamma=2,b=1,**kwargs):
    """
    ECA-NET
    :param inputs_tensor: input_tensor.shape=[batchsize,h,w,channels]
    :param num:
    :param gamma:
    :param b:
    :return:
    """
    channels = K.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels,2)+b)/gamma))
    k = t if t%2 else t+1
    x_global_avg_pool = GlobalAveragePooling2D()(inputs_tensor)
    x = Reshape((channels,1))(x_global_avg_pool)
    x = Conv1D(1,kernel_size=k,padding="same",name="eca_conv1_" + str(num))(x)
    x = Activation('sigmoid', name='eca_conv1_relu_' + str(num))(x)  #shape=[batch,chnnels,1]
    x = Reshape((1, 1, channels))(x)
    output = multiply([inputs_tensor,x])
    return output

def model(input_shape):
    H = input_shape[0]
    W = input_shape[1]
    def downsample_block(x, block_num, n_filters, pooling_on=True):

        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv1")(x)
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv2")(x)
        x = eca_layer(inputs_tensor=x, num=block_num)
        skip = x

        if pooling_on is True:
            x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name="Block" + str(block_num) + "_Pool1")(x)

        return x, skip

    def upsample_block(x, skip, block_num, n_filters):

        x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=2, padding='valid', activation='relu',
                            name="Block" + str(block_num) + "_ConvT1")(x)
        x = concatenate([x, skip], axis=-1, name="Block" + str(block_num) + "_Concat1")
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv1")(x)
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv2")(x)
        x = eca_layer(inputs_tensor=x, num=block_num)

        return x

    input = Input(input_shape, name="Input")

    # downsampling
    x, skip1 = downsample_block(input, 1, 64)
    x, skip2 = downsample_block(x, 2, 128)
    x, skip3 = downsample_block(x, 3, 256)
    x, skip4 = downsample_block(x, 4, 512)
    x, _ = downsample_block(x, 5, 1024, pooling_on=False)

    # upsampling
    x = upsample_block(x, skip4, 6, 512)
    x = upsample_block(x, skip3, 7, 256)
    x = upsample_block(x, skip2, 8, 128)
    x = upsample_block(x, skip1, 9, 64)

    output = Conv2D(15, kernel_size=(1, 1), strides=1, padding='valid', activation='linear', name="output")(x)
    #output = Reshape(target_shape=(H*W*Nkeypoints,1))(output)#add and check
    model = Model(inputs=input, outputs=output, name="Output")
    model.summary()
    return model
if __name__=="__main__":
    model(input_shape)

