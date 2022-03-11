import tensorflow.keras.backend as K
import json
from tqdm import tqdm
import random
import glob
import pandas as pd
import os
import cv2
import tensorflow.keras
import numpy as np
#from fcn import model
from hourglass import model
from matplotlib import pyplot as plt
#from load_numpy_data2 import generator
#from load_numpy_data_heatmap96x96 import generator# 1933912,1
#from load_numpy_data_face import generator
from load_numpy_data_face_multistage import generator
from segmentation_metrics import mean_iou, mean_dice

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
input_shape = (96, 96, 1)
#input_shape = (256, 256, 3)
def get_model():
    #return model(input_shape)
    return model(num_classes=15, num_stacks=5, num_channels=128, model_input_shape=input_shape)#hourglass
    #return model(input=input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)

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
model = get_model()
#model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])#original
#model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])#original
#model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])#original
#model.compile(optimizer="adam", loss=jaccard, metrics=["accuracy"])
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.load_weights("hourglassfacemultistage52.hdf5")
#model.load_weights("test.hdf5") #Accuracy: 91.47552847862244 % Mean IOU : 0.561591
#model.load_weights("unet2mse.h5") #Accuracy: 86.97680234909058 % Mean IOU : 0.4135831

#loss, loss1,loss2, loss3,loss4, loss5,acc1, acc2,acc3, acc4, acc5 = model.evaluate(train_generator)#model.evaluate_generator
#loss, loss1,loss2, loss3,loss4, loss5,acc1, acc2,acc3, acc4, acc5 = model.evaluate(validation_generator)#model.evaluate_generator


def findCoordinates(mask):

    hm_sum = np.sum(mask)

    index_map = [j for i in range(96) for j in range(96)]
    index_map = np.reshape(index_map, newshape=(96,96))

    x_score_map = mask * index_map / hm_sum
    y_score_map = mask * np.transpose(index_map) / hm_sum

    px = np.sum(np.sum(x_score_map, axis=None))
    py = np.sum(np.sum(y_score_map, axis=None))

    return px, py


def showAllMasks(img_mask, nrows=3, ncols=5):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    r = -1

    for i in range(img_mask.shape[-1]):

        img = img_mask[:, :, i]
        # print(img.shape)
        img = np.reshape(img, newshape=(96,96))
        img = np.stack([img,img,img], axis=-1)

        c = i % ncols

        if i % ncols == 0:
            r += 1

        axes[r, c].imshow(img)

    plt.show()

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

iou_total=0
dice_total=0
for train_x, train_y in tqdm(train_generator):
    print(train_x.shape, train_y[0].shape)
    print(train_x[0].shape, train_y[0][0].shape)#train_y[stage][index/batch]
    y_pred = model.predict(train_x) #y_pred[stage][index]
    iou = mean_iou(train_y[4], y_pred[4])
    dice = mean_dice(train_y[4], y_pred[4])
    # iou_total.append(iou)
    # dice_total.append(dice)
    iou_total = iou_total + iou
    dice_total = dice_total + dice
    break
print("train mean iou is", iou_total / len(train_generator))
print("train mean dice is", dice_total / len(train_generator))

# iou_total=0
# dice_total=0
# for val_x, val_y in tqdm(validation_generator):
#     print(val_x.shape, val_y[0].shape)
#     print(val_x[0].shape, val_y[0][0].shape)
#     y_pred = model.predict(val_x)
#     iou = mean_iou(val_y[4], y_pred[4])
#     dice = mean_dice(val_y[4], y_pred[4])
#     # iou_total.append(iou)#list is [1 2 3....]
#     # dice_total.append(dice)
#     iou_total = iou_total + iou
#     dice_total = dice_total + dice
#     break
# # iou_total = np.array(iou_total)#array is [1, 2, 3,....]
# # dice_total = np.array(dice_total)
# print("val mean iou is", iou_total / len(validation_generator))
# print("val mean dice is", dice_total / len(validation_generator))



index = 15
plt.imshow(train_x[index], cmap ="gray")  # 368,368 for 368,368,21
plt.axis('off')
plt.savefig("input_image.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.array(train_y[0][index]).sum(axis=-1),cmap="jet")  # same
plt.axis('off')
plt.savefig("input_heatmap.png", bbox_inches='tight', pad_inches=0)
plt.show()


gt_list = []
pred_list = []

for k in range(15):
    xpred, ypred = findCoordinates(np.array(train_y[0][index])[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list.append(xpred)
    pred_list.append(ypred)
pred_list = np.array(pred_list, dtype=np.float32)
print(pred_list.shape)

#plot_keypoints(train_x[index], pred_list)
#plot_keypoints(train_x[0], pred_list)
for i in range(0, 30, 2):
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("input_keypoints.png", bbox_inches='tight', pad_inches=0)
plt.show()

#y_pred = model.predict(train_x)#
print(y_pred[0].shape)#batch, 96x96x21, 1
plt.imshow(np.array(y_pred[0][index]).sum(axis=-1),cmap="jet")  # 96,96 #y_pred[stage output num][index/batch num]
print(np.array(y_pred[0][index]).sum(axis=-1).shape)#96,96
plt.axis('off')
plt.savefig("output_heatmap_1.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.array(y_pred[1][index]).sum(axis=-1),cmap="jet")  # 96,96
print(np.array(y_pred[1][index]).sum(axis=-1).shape)#96,96
plt.axis('off')
plt.savefig("output_heatmap_2.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.array(y_pred[2][index]).sum(axis=-1),cmap="jet")  # 96,96
print(np.array(y_pred[2][index]).sum(axis=-1).shape)#96,96
plt.axis('off')
plt.savefig("output_heatmap_3.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.array(y_pred[3][index]).sum(axis=-1),cmap="jet")  # 96,96
print(np.array(y_pred[3][index]).sum(axis=-1).shape)#96,96
plt.axis('off')
plt.savefig("output_heatmap_4.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.array(y_pred[4][index]).sum(axis=-1),cmap="jet")  # 96,96
print(np.array(y_pred[4][index]).sum(axis=-1).shape)#96,96
plt.axis('off')
plt.savefig("output_heatmap_5.png", bbox_inches='tight', pad_inches=0)
plt.show()

#individual masks
#showAllMasks(y_pred[0][index])
#showAllMasks(y_pred[1][index])
#showAllMasks(y_pred[2][index])
#showAllMasks(y_pred[3][index])
#showAllMasks(y_pred[4][index])

gt_list = []
pred_list = []
pred_list2 = []
pred_list3 = []
pred_list4 = []
pred_list5 = []

for k in range(15):
    xpred, ypred = findCoordinates(np.array(y_pred[0][index])[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list.append(xpred)
    pred_list.append(ypred)
pred_list = np.array(pred_list, dtype=np.float32)
print(pred_list.shape)

#plot_keypoints(train_x[index], pred_list)
for i in range(0, 30, 2):
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("output_keypoints_1.png", bbox_inches='tight', pad_inches=0)
plt.show()

for k in range(15):
    xpred, ypred = findCoordinates(np.array(y_pred[1][index])[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list2.append(xpred)
    pred_list2.append(ypred)
pred_list2 = np.array(pred_list2, dtype=np.float32)
print(pred_list2.shape)

#plot_keypoints(train_x[index], pred_list2)
for i in range(0, 30, 2):
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("output_keypoints_2.png", bbox_inches='tight', pad_inches=0)
plt.show()

for k in range(15):
    xpred, ypred = findCoordinates(np.array(y_pred[2][index])[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list3.append(xpred)
    pred_list3.append(ypred)
pred_list3 = np.array(pred_list3, dtype=np.float32)
print(pred_list3.shape)

#plot_keypoints(train_x[index], pred_list3)
for i in range(0, 30, 2):
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("output_keypoints_3.png", bbox_inches='tight', pad_inches=0)
plt.show()

for k in range(15):
    xpred, ypred = findCoordinates(np.array(y_pred[3][index])[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list4.append(xpred)
    pred_list4.append(ypred)
pred_list4 = np.array(pred_list4, dtype=np.float32)
print(pred_list4.shape)

#plot_keypoints(train_x[index], pred_list4)
for i in range(0, 30, 2):
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("output_keypoints_4.png", bbox_inches='tight', pad_inches=0)
plt.show()

for k in range(15):
    xpred, ypred = findCoordinates(np.array(y_pred[4][index])[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list5.append(xpred)
    pred_list5.append(ypred)
pred_list5 = np.array(pred_list5, dtype=np.float32)
print(pred_list5.shape)

#plot_keypoints(train_x[index], pred_list5)
for i in range(0, 30, 2):
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("output_keypoints_5.png", bbox_inches='tight', pad_inches=0)
plt.show()