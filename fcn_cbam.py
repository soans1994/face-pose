from tensorflow.keras.layers import Input, Conv2D, Reshape, MaxPooling2D, Conv2DTranspose, Add,Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.regularizers import l2

input_shape = (96,96,1)
#input_shape = (256,256,1)
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

    def conv_block(x, nconvs, n_filters, block_name, wd=None):
        for i in range(nconvs):
            x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                       kernel_regularizer=wd, name=block_name + "_conv" + str(i + 1))(x)

        x = ChannelAttentionModule(x)
        x = SpatialAttentionModule(x)
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

