from tensorflow.keras.layers import Input, concatenate, Dense, Flatten, BatchNormalization, Dropout, ReLU, Conv2D, Reshape, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.regularizers import l2

input_shape = (96,96,1)
Nkeypoints = 15
def ChannelAttentionModule(input: tf.keras.Model, ratio=16):
    channel = input.shape[-1]

    shared_dense_one = tf.keras.layers.Dense(channel // ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
    shared_dense_two = tf.keras.layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    x = tf.keras.layers.Add()([avg_pool, max_pool])
    x = tf.keras.layers.Activation('sigmoid')(x)

    return tf.keras.layers.multiply([input, x])

def SpatialAttentionModule(input: tf.keras.Model, kernel_size=3):
    avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input)
    max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input)
    x = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    for i in [64, 32, 16]:
        x = tf.keras.layers.Conv2D(filters=i,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer='he_normal',
                                   use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filters=1,
                               kernel_size=kernel_size,
                               strides=1,
                               padding='same',
                               activation='sigmoid',
                               kernel_initializer='he_normal',
                               use_bias=False)(x)

    return tf.keras.layers.multiply([input, x])

def model(input_shape):
    H = input_shape[0]
    W = input_shape[1]
    def downsample_block(x, block_num, n_filters, pooling_on=True):

        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv1")(x)
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                   name="Block" + str(block_num) + "_Conv2")(x)
        x = ChannelAttentionModule(x)
        x = SpatialAttentionModule(x)
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
        x = ChannelAttentionModule(x)
        x = SpatialAttentionModule(x)

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

    output = Conv2D(16, kernel_size=(1, 1), strides=1, padding='valid', activation='softmax', name="output")(x)
    #output = Reshape(target_shape=(H*W*Nkeypoints,1))(output)#add and check
    model = Model(inputs=input, outputs=output, name="Output")
    model.summary()
    return model
if __name__=="__main__":
    model(input_shape)

