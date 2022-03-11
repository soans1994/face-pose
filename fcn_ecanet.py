from tensorflow.keras.layers import Input, Conv2D, Reshape, MaxPooling2D, Conv2DTranspose, Add, GlobalAveragePooling2D, Conv1D, Activation, multiply
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import tensorflow as tf
import math
from tensorflow.keras.regularizers import l2

input_shape = (96,96,1)
#input_shape = (256,256,1)
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

    def conv_block(x, nconvs, n_filters, block_name, wd=None):
        for i in range(nconvs):
            x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                       kernel_regularizer=wd, name=block_name + "_conv" + str(i + 1))(x)
        x = eca_layer(inputs_tensor=x, num=block_name)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name=block_name + "_pool")(x)
        #x = Dropout(0.2)(x)#added to check improve val acc---no improve
        return x

    input = Input(shape=input_shape, name="Input")

    # Block 1
    x = conv_block(input, nconvs=2, n_filters=64, block_name="block1")

    # Block 2
    x = conv_block(x, nconvs=2, n_filters=128, block_name="block2")

    # Block 3
    pool3 = conv_block(x, nconvs=3, n_filters=256, block_name="block3")

    # Block 4
    pool4 = conv_block(pool3, nconvs=3, n_filters=512, block_name="block4")

    # Block 5
    x = conv_block(pool4, nconvs=3, n_filters=512, block_name="block5")

    # convolution 6
    x = Conv2D(4096, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv6")(x)
    # convolution 7
    x = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv7")(x)
    # upsampling
    preds_pool3 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool3")(pool3)
    preds_pool4 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool4")(pool4)
    up_pool4 = Conv2DTranspose(filters=15, kernel_size=2, strides=2, activation="relu", name="ConvT_pool4")(preds_pool4)
    up_conv7 = Conv2DTranspose(filters=15, kernel_size=4, strides=4, activation="relu", name="ConvT_conv7")(x)
    fusion = Add()([preds_pool3, up_pool4, up_conv7])

    output = Conv2DTranspose(filters=15, kernel_size=8, strides=8, activation='relu', name="convT_fusion")(fusion)
    output = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="linear", name="output")(output)
    #output = Reshape(target_shape=(H * W * Nkeypoints, 1))(output)

    model = Model(inputs=input, outputs=output, name="FCN8")
    model.summary()
    return model
if __name__=="__main__":
    model(input_shape)

