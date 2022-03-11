from tensorflow.keras.layers import Input, concatenate, Permute, Flatten, BatchNormalization, Dropout, ReLU, Conv2D, Reshape, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Reshape,MaxPooling2D, Conv2DTranspose, Add,Dense, GlobalAveragePooling2D, multiply
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.regularizers import l2

input_shape = (96,96,1)
Nkeypoints = 15

def squeeze_excite_block(tensor, n_filters, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = n_filters
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def model(input_shape):
    H = input_shape[0]
    W = input_shape[1]
    def downsample_block(x, block_num, n_filters, pooling_on=True):

        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv1")(x)
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv2")(x)
        x = squeeze_excite_block(x, n_filters)
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
        x = squeeze_excite_block(x, n_filters)

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

