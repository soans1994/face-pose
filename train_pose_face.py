import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
import pandas as pd
import random
import os
import math
from unet import model
#from hrnet import model

from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
#from load_numpy_data_face import generator # same
#from load_numpy_data_face_backup import generator # same
#from load_numpy_data_face_color_segmentation_mask import generator
#from load_numpy_data_face_color_segmentation_mask_categorical import generator
from load_coco_face import load_and_filter_annotations
from data_generator import generator2, generator3, generator3gray #generator2 single , generator3 multiple
from data_generator import generator22#, generator33  # binary mask generator22 single , generator33 multiple

# plots keypoints on face image
def plot_keypoints(img, points):
    # display image
    plt.imshow(img, cmap='gray')
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 30, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(points[i], points[i + 1], color='red')
        # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    plt.show()

train_annot_path = 'coco/annotations/coco_wholebody_train_v1.0.json'
val_annot_path = 'coco/annotations/coco_wholebody_val_v1.0.json'
train_img_path = 'coco/' #automatic img path from dataframe path
val_img_path = 'coco/'

train_df, val_df = load_and_filter_annotations(train_annot_path, val_annot_path, subset=1.0)

train_generator = generator22(train_df, train_img_path, shuffle=True, batch_size=256, input_dim=(96,96), output_dim=(96,96))
validation_generator = generator22(val_df, val_img_path, shuffle=True, batch_size=256, input_dim=(96,96), output_dim=(96,96))

print(len(train_generator), len(validation_generator))
#train_generator = DataGenerator(train_images, train_labels, batch_size = 32)#119 batches(3824/32)
#validation_generator = DataGenerator(test_images, test_labels, batch_size = 32)
for i,j in train_generator:
    print(i.shape, j.shape)
    print(i[0].shape, j[0].shape)
    break
id = 1#15
plot_keypoints(i[id], j[id])

#input_shape = (256, 256, 3)
input_shape = (96, 96, 3)
print(input_shape)

def mean_squared_error2(y_true, y_pred):
    channel_loss = K.sum(K.square(y_pred - y_true), axis=-1)
    total_loss = K.mean(channel_loss, axis=-1)
    print(total_loss.shape)
    return total_loss
def jaccard(ytrue, ypred, smooth=1e-5):
    intersection = K.sum(K.abs(ytrue*ypred), axis=-1)
    union = K.sum(K.abs(ytrue)+K.abs(ypred), axis=-1)
    jac = (intersection + smooth) / (union-intersection+smooth)
    return K.mean(jac)
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice_coef = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return 1 - dice_coef
def get_model():
    return model(input_shape)
    #return model(16, 96, 96, 1, 15)#hrnet
    #return model(input_shape=input_shape, num_classes=num_classes)
    #return model(input=input_shape, num_classes=num_classes)
model = get_model()
#model.load_weights("vgg16s3.hdf5")
#optimizer = tf.keras.optimizers.Adam(0.1)
#model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])  # default lr 0.001,1e-3
#model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])  # default lr 0.001,1e-3
#model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["accuracy"]) #original adam 16% 9%
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]) #original adam 16% 9%
#model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss="mse", metrics=["accuracy"]) #rmsprop mse 17% 12, 20, 11
#model.compile(optimizer=keras.optimizers.SGD(1e-3), loss="mean_squared_error", metrics=["accuracy"]) #sgd mse
#model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss=jaccard, metrics=["accuracy"]) # rmsprop 2%
#model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=jaccard, metrics=["accuracy"]) #regression
#model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=[tf.keras.metrics.RootMeanSquaredError()])  # default lr 0.001,1e-3# original
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # default lr 0.001,1e-3

#lr = 1e-3
callbacks = [ModelCheckpoint("test.hdf5", verbose=1, save_best_only=True),
             #ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1, min_lr=1e-6, ),# sdnt go below min_lr
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001, verbose=5),
            CSVLogger("results.csv"),
             EarlyStopping(monitor="val_loss", patience=10, verbose=1)]# try val_root_mean_squared_error
# history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs= 500, validation_data=(x_test, y_test), shuffle=False) #callbacks=callbacks)#, class_weight=class_weights )
#history = model.fit(train_images, train_labels, batch_size=32, verbose=1, epochs=300, validation_split=0.3,shuffle=False, callbacks=callbacks)#
history = model.fit(train_generator, verbose=1, epochs=500, validation_data=validation_generator, shuffle=True, callbacks=callbacks)#
# history = model.fit(x_train, y_train_cat, batch_size=2, verbose=1, epochs= 10, validation_data=(x_test, y_test_cat), shuffle=False)#, class_weight=class_weights )
# shuffle true sshuffles only the training data for every epoch. but may be we need same for checking imporved models.
#model.save("test.hdf5")


#_, acc = model.evaluate(test_images, test_labels)
_, acc = model.evaluate(validation_generator)
print("Accuracy of test set:", (acc * 100.0), "%")
#_, acc = model.evaluate(train_images, train_labels)
_, acc = model.evaluate(train_generator)
print("Accuracy of train set:", (acc * 100.0), "%")

# plot train val acc loss

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "y", label="loss")
plt.plot(epochs, val_loss, "r", label="val loss")
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig("loss1.png")
plt.show()

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, "y", label="Training loss")
# plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.plot(epochs, loss, color="#1f77b4", label="loss")
plt.plot(epochs, val_loss, color="#ff7f0e", label="val loss")
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig("loss.png")
plt.show()
plt.close()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
# plt.plot(epochs, acc, "y", label="Training Accuracy")
# plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.plot(epochs, acc, color="#1f77b4", label="acc")
plt.plot(epochs, val_acc, color="#ff7f0e", label="val acc")
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid()
plt.legend()
plt.savefig("accuracy.png")
plt.show()
plt.close()

"""
fig = plt.figure(figsize=(15, 15))
# make test images keypoints prediction
points_test = model.predict(test_images)
points_train = model.predict(train_images)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(test_images[i], np.squeeze(points_test[i]))
    #plot_keypoints(test_images[i], points_test[i])
    plt.show()
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_keypoints(train_images[i], np.squeeze(points_train[i]))
    #plot_keypoints(train_images[i], points_train[i])
    plt.show()
"""
a = 1
