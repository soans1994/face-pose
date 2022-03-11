import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error
import pandas as pd
import os
import cv2
import tensorflow.keras
import numpy as np
from tqdm import tqdm
#from unet_avgpool import model
#from unet import model
#from hourglass import model
#from hrnet import model
#from unet_ecanet import model
#from unet import model
from unet_cbam import model
#from unet_senet3 import model
#from unet_ecanet2 import model
#from fcn import model
#from unetfinal import model
#from hrnet import model
from matplotlib import pyplot as plt
#from load_numpy_data2 import generator
#from load_numpy_data_heatmap96x96 import generator# 1933912,1
#from load_numpy_data_face import generator
#from load_numpy_data_face_noshuffle import generator
#from load_numpy_data_face_color import generator
#from load_numpy_data_face_color_segmentation_mask import generator
from load_numpy_data_face_color_segmentation_mask_categorical import generator
from segmentation_metrics import mean_iou, mean_dice
from segmentation_metrics import mean_iou2, mean_dice2
from segmentation_metrics import l2_metric, pck_metric, multi_thresh_pck


print(len(train_generator), len(validation_generator))
input_shape = (256, 256, 3)
def get_model():
    return model(input_shape)
    #return model(16, 96, 96, 1, 15)#hrnet
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
def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice_coef = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return 1 - dice_coef
model = get_model()
#model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])#original
#model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])#original
#model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])#original
#model.compile(optimizer="adam", loss=jaccard, metrics=["accuracy"])
#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
#model.load_weights("generator_noadv_unet_camvid.h5")
model.load_weights("generator_adv_unet_camvid.h5")


index = 1

iou_total=0
dice_total=0
for train_x, train_y in tqdm(train_generator):
    #print(train_x.shape, train_y.shape)#(16, 96, 96, 1) (16, 96, 96, 15)
    #print(train_x[index].shape, train_y[index].shape)#(96, 96, 1) (96, 96, 15)
    y_pred_train = model.predict(train_x)  #
    #print(y_pred.shape)  # batch, 96x96x15, 1
    iou = mean_iou(train_y, y_pred_train)
    dice = mean_dice(train_y, y_pred_train)
    #iou_total.append(iou)
    #dice_total.append(dice)
    iou_total = iou_total + iou
    dice_total = dice_total + dice
    #break
print("train mean iou is", iou_total/len(train_generator))
print("train mean dice is", dice_total/len(train_generator))


iou_total=0
dice_total=0
for val_x, val_y in tqdm(validation_generator):
    #print(val_x.shape, val_y.shape)
    #print(val_x[index].shape, val_y[index].shape)
    y_pred_val = model.predict(val_x)  #
    #print(y_pred.shape)  # batch, 96x96x21, 1
    iou = mean_iou(val_y, y_pred_val)
    dice = mean_dice(val_y, y_pred_val)
    #iou_total.append(iou)#list is [1 2 3....]
    #dice_total.append(dice)
    iou_total = iou_total + iou
    dice_total = dice_total + dice
    #break
#iou_total = np.array(iou_total)#array is [1, 2, 3,....]
#dice_total = np.array(dice_total)
print("val mean iou is", iou_total / len(validation_generator))
print("val mean dice is", dice_total / len(validation_generator))


iou_total=0
dice_total=0
for val_x, val_y in tqdm(validation_generator):
    #print(val_x.shape, val_y.shape)
    #print(val_x[index].shape, val_y[index].shape)
    y_pred_val = model.predict(val_x)  #
    #print(y_pred.shape)  # batch, 96x96x21, 1
    iou = mean_iou2(val_y, y_pred_val)
    dice = mean_dice2(val_y, y_pred_val)
    #iou_total.append(iou)#list is [1 2 3....]
    #dice_total.append(dice)
    iou_total = iou_total + iou
    dice_total = dice_total + dice
    #break
#iou_total = np.array(iou_total)#array is [1, 2, 3,....]
#dice_total = np.array(dice_total)
print("val classwise iou is", iou_total / len(validation_generator))
print("val classwise dice is", dice_total / len(validation_generator))
print("K.mean iou is", K.mean(iou_total / len(validation_generator)))
print("K.mean dice is", K.mean(dice_total / len(validation_generator)))







