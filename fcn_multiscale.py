from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, Add,AveragePooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow as tf
from tensorflow.keras.regularizers import l2

input_shape1 = (96,96,1)
input_shape2 = (192,192,1)
input_shape3 = (288,288,1)
#input_shape = (256,256,1)
Nkeypoints = 15

def model(input_shape1,input_shape2, input_shape3):
    H = input_shape1[0]
    W = input_shape1[1]

    def conv_block(x, nconvs, n_filters, block_name, wd=None):
        for i in range(nconvs):
            x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                       kernel_regularizer=wd, name=block_name + "_conv" + str(i + 1))(x)

        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name=block_name + "_pool")(x)
        #x = Dropout(0.2)(x)#added to check improve val acc---no improve
        return x

    input1 = Input(shape=input_shape1, name="Input1")
    input2 = Input(shape=input_shape2, name="Input2")
    input3 = Input(shape=input_shape3, name="Input3")

    # Block 1
    x1 = conv_block(input1, nconvs=2, n_filters=64, block_name="block11")
    x2 = conv_block(input2, nconvs=2, n_filters=64, block_name="block12")
    x3 = conv_block(input3, nconvs=2, n_filters=64, block_name="block13")

    # Block 2
    x1 = conv_block(x1, nconvs=2, n_filters=128, block_name="block21")
    x2 = conv_block(x2, nconvs=2, n_filters=128, block_name="block22")
    x3 = conv_block(x3, nconvs=2, n_filters=128, block_name="block23")

    # Block 3
    pool31 = conv_block(x1, nconvs=3, n_filters=256, block_name="block31")
    pool32 = conv_block(x2, nconvs=3, n_filters=256, block_name="block32")
    pool33 = conv_block(x3, nconvs=3, n_filters=256, block_name="block33")

    # Block 4
    pool41 = conv_block(pool31, nconvs=3, n_filters=512, block_name="block41")
    pool42 = conv_block(pool32, nconvs=3, n_filters=512, block_name="block42")
    pool43 = conv_block(pool33, nconvs=3, n_filters=512, block_name="block43")

    # Block 5
    x1 = conv_block(pool41, nconvs=3, n_filters=512, block_name="block51")
    x2 = conv_block(pool42, nconvs=3, n_filters=512, block_name="block52")
    x3 = conv_block(pool43, nconvs=3, n_filters=512, block_name="block53")

    # convolution 6
    x1 = Conv2D(4096, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv61")(x1)
    x2 = Conv2D(4096, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv62")(x2)
    x3 = Conv2D(4096, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv63")(x3)
    # convolution 7
    x1 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv71")(x1)
    x2 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv72")(x2)
    x3 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="relu", name="conv73")(x3)
    # upsampling
    preds_pool31 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool31")(pool31)
    preds_pool32 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool32")(pool32)
    preds_pool33 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool33")(pool33)
    preds_pool41 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool41")(pool41)
    preds_pool42 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool42")(pool42)
    preds_pool43 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", name="preds_pool43")(pool43)
    up_pool41 = Conv2DTranspose(filters=15, kernel_size=2, strides=2, activation="relu", name="ConvT_pool41")(preds_pool41)
    up_pool42 = Conv2DTranspose(filters=15, kernel_size=2, strides=2, activation="relu", name="ConvT_pool42")(preds_pool42)
    up_pool43 = Conv2DTranspose(filters=15, kernel_size=2, strides=2, activation="relu", name="ConvT_pool43")(preds_pool43)
    up_conv71 = Conv2DTranspose(filters=15, kernel_size=4, strides=4, activation="relu", name="ConvT_conv71")(x1)
    up_conv72 = Conv2DTranspose(filters=15, kernel_size=4, strides=4, activation="relu", name="ConvT_conv72")(x2)
    up_conv73 = Conv2DTranspose(filters=15, kernel_size=4, strides=4, activation="relu", name="ConvT_conv73")(x3)
    fusion1 = Add()([preds_pool31, up_pool41, up_conv71])
    fusion2 = Add()([preds_pool32, up_pool42, up_conv72])
    fusion3 = Add()([preds_pool33, up_pool43, up_conv73])

    output1 = Conv2DTranspose(filters=15, kernel_size=8, strides=8, activation='relu', name="convT_fusion1")(fusion1)
    output2 = Conv2DTranspose(filters=15, kernel_size=8, strides=8, activation='relu', name="convT_fusion2")(fusion2)
    output3 = Conv2DTranspose(filters=15, kernel_size=8, strides=8, activation='relu', name="convT_fusion3")(fusion3)
    output1 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="linear", name="output1")(output1)
    output2 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="linear", name="output2")(output2)
    output3 = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="linear", name="output3")(output3)
    output2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output2)
    output3 = MaxPooling2D(pool_size=(4, 4), strides=3, padding='same')(output3)
    output = Concatenate()([output1, output2, output3])
    #output = Conv2D(15, kernel_size=(1, 1), strides=1, padding="same", activation="linear", name="output")(output)
    #output = Reshape(target_shape=(H * W * Nkeypoints, 1))(output)
    model = Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3], name="FCN8")
    model.summary()
    return model
if __name__=="__main__":
    model(input_shape1,input_shape2, input_shape3)

