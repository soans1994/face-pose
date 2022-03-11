import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
import pandas as pd
import random
import os
import math
#from unet import model
from fcn_multistage53 import model
#from unet_multistage5 import model
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from load_numpy_data_face_multistage import generator

# plots keypoints on face image
def plot_keypoints(img, points):
    # display image
    plt.imshow(img, cmap='gray')
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 42, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(points[i], points[i + 1], color='red')
        # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    plt.show()

data_dir = "face"
train_dir = "train"
train_csv = "training.csv"
test_dir = "test"
test_csv = "test.csv"

df_train = pd.read_csv(os.path.join(data_dir, train_csv))
df_test = pd.read_csv(os.path.join(data_dir, test_csv))

n_train = df_train['Image'].size
n_test = df_test['Image'].size

df_kp = df_train.iloc[:,0:30]

idxs = []

img_dict = {}
kp_dict = {}

for i in range(n_train):

    if True in df_train.iloc[i, 0:30].isna().values:
        continue
    else:
        idxs.append(i)

        img_dict[i] = "train"+str(i)+".png"

        # keypoints
        kp = df_kp.iloc[i].values.tolist()
        kp_dict[i] = kp

random.shuffle(idxs)

# subset = int(0.1*len(idxs))

cutoff_idx = int(0.9*len(idxs))
train_idxs = idxs[0:cutoff_idx]
val_idxs = idxs[cutoff_idx:len(idxs)]

print("\n# of Training Images: {}".format(len(train_idxs)))
print("# of Val Images: {}".format(len(val_idxs)))

transform_dict = {"Flip": False, "Shift": False, "Scale": False, "Rotate": False}

train_generator = generator(os.path.join(data_dir, train_dir),
                              train_idxs,
                              img_dict,
                              kp_dict,
                              transform_dict=transform_dict,
                              augment=False,
                              batch_size=16)

validation_generator = generator(os.path.join(data_dir, train_dir),
                            val_idxs,
                            img_dict,
                            kp_dict,
                            augment=False,
                            batch_size=16)

print(len(train_generator), len(validation_generator))
#train_generator = DataGenerator(train_images, train_labels, batch_size = 32)#119 batches(3824/32)
#validation_generator = DataGenerator(test_images, test_labels, batch_size = 32)
for i,j in train_generator:
    print(i.shape, j[0].shape)
    print(i[0].shape, j[0].shape)
    break
id = 1#15
plot_keypoints(i[id], j[0][id])

#input_shape = (368, 368, 3)
#input_shape = (256, 256, 1)
input_shape = (96, 96, 1)
#input_shape = (256, 256, 3)
num_classes = 30
Nkeypoints = 15
print(input_shape)

def get_loss_func():
    def mse(x, y):
        return mean_squared_error(x, y)

    keys = ['output_stage1', 'output_stage2', 'output_stage3', 'output_stage4', 'output_stage5']
    losses = dict.fromkeys(keys, mse)
    return losses
losses = get_loss_func()

def get_model():
    return model(input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)
    #return model(input=input_shape, num_classes=num_classes)

model = get_model()
#model.load_weights("vgg16s3.hdf5")
#optimizer = tf.keras.optimizers.Adam(0.1)
#model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"])  # default lr 0.001,1e-3
#model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])  # default lr 0.001,1e-3
#model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["accuracy"]) #original adam 16% 9%
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses, metrics=["accuracy"]) #original adam 16% 9%
#model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss="mse", metrics=["accuracy"]) #rmsprop mse 17% 12, 20, 11
#model.compile(optimizer=keras.optimizers.SGD(1e-3), loss="mse", metrics=["accuracy"]) #sgd mse
#model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss=jaccard, metrics=["accuracy"]) # rmsprop 2%
#model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=jaccard, metrics=["accuracy"]) #regression
#model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])  # default lr 0.001,1e-3# original
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # default lr 0.001,1e-3

#lr = 1e-3
callbacks = [ModelCheckpoint("test.hdf5", verbose=1, save_best_only=True),
             #ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1, min_lr=1e-6, ),# sdnt go below min_lr
             EarlyStopping(monitor="val_loss", patience=10, verbose=1)]# try val_root_mean_squared_error
# history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs= 500, validation_data=(x_test, y_test), shuffle=False) #callbacks=callbacks)#, class_weight=class_weights )
#history = model.fit(train_images, train_labels, batch_size=32, verbose=1, epochs=300, validation_split=0.3,shuffle=False, callbacks=callbacks)#
history = model.fit(train_generator, verbose=1, epochs=500, validation_data=validation_generator, shuffle=True, callbacks=callbacks)#
# history = model.fit(x_train, y_train_cat, batch_size=2, verbose=1, epochs= 10, validation_data=(x_test, y_test_cat), shuffle=False)#, class_weight=class_weights )
# shuffle true sshuffles only the training data for every epoch. but may be we need same for checking imporved models.
#model.save("test.hdf5")


#_, acc = model.evaluate(test_images, test_labels)
#_, acc = model.evaluate(validation_generator)
#print("Accuracy of test set:", (acc * 100.0), "%")
#_, acc = model.evaluate(train_images, train_labels)
#_, acc = model.evaluate(train_generator)
#print("Accuracy of train set:", (acc * 100.0), "%")

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
plt.plot(epochs, loss, label="loss")
plt.plot(epochs, val_loss, label="val loss")
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig("loss.png")
plt.show()
plt.close()

stage1_loss = history.history["output_stage1_loss"]
stage2_loss = history.history["output_stage2_loss"]
stage3_loss = history.history["output_stage3_loss"]
stage4_loss = history.history["output_stage4_loss"]
stage5_loss = history.history["output_stage5_loss"]
val_stage1_loss = history.history["val_output_stage1_loss"]
val_stage2_loss = history.history["val_output_stage2_loss"]
val_stage3_loss = history.history["val_output_stage3_loss"]
val_stage4_loss = history.history["val_output_stage4_loss"]
val_stage5_loss = history.history["val_output_stage5_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, stage1_loss, label="stage1 loss")
plt.plot(epochs, stage2_loss, label="stage2 loss")
plt.plot(epochs, stage3_loss, label="stage3 loss")
plt.plot(epochs, stage4_loss, label="stage4 loss")
plt.plot(epochs, stage5_loss, label="stage5 loss")
plt.plot(epochs, val_stage1_loss, label="val_stage1 loss")
plt.plot(epochs, val_stage2_loss, label="val_stage2 loss")
plt.plot(epochs, val_stage3_loss, label="val_stage3 loss")
plt.plot(epochs, val_stage4_loss, label="val_stage4 loss")
plt.plot(epochs, val_stage5_loss, label="val_stage5 loss")
plt.title("model stage loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig("stage_loss.png")
plt.show()
plt.close()

stage1_acc = history.history["output_stage1_accuracy"]
stage2_acc = history.history["output_stage2_accuracy"]
stage3_acc = history.history["output_stage3_accuracy"]
stage4_acc = history.history["output_stage4_accuracy"]
stage5_acc = history.history["output_stage5_accuracy"]
val_stage1_acc = history.history["val_output_stage1_accuracy"]
val_stage2_acc = history.history["val_output_stage2_accuracy"]
val_stage3_acc = history.history["val_output_stage3_accuracy"]
val_stage4_acc = history.history["val_output_stage4_accuracy"]
val_stage5_acc = history.history["val_output_stage5_accuracy"]
# plt.plot(epochs, acc, "y", label="Training Accuracy")
# plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.plot(epochs, stage1_acc, label="stage1 acc")
plt.plot(epochs, stage2_acc, label="stage2 acc")
plt.plot(epochs, stage3_acc, label="stage3 acc")
plt.plot(epochs, stage4_acc, label="stage4 acc")
plt.plot(epochs, stage5_acc, label="stage5 acc")
plt.plot(epochs, val_stage1_acc, label="val_stage1 acc")
plt.plot(epochs, val_stage2_acc, label="val_stage2 acc")
plt.plot(epochs, val_stage3_acc, label="val_stage3 acc")
plt.plot(epochs, val_stage4_acc, label="val_stage4 acc")
plt.plot(epochs, val_stage5_acc, label="val_stage5 acc")
plt.title("model stage accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid()
plt.legend()
plt.savefig("stage_accuracy.png")
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
