import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error
import pandas as pd
import os
import cv2
from PIL import Image
import tensorflow.keras
import numpy as np

from unet import model

from matplotlib import pyplot as plt


def findCoordinates(mask):

    hm_sum = np.sum(mask) #repeats 21 times for each keypoint

    index_map = [j for i in range(96) for j in range(96)]
    index_map = np.reshape(index_map, newshape=(96,96))

    x_score_map = mask * index_map / hm_sum #96x96
    y_score_map = mask * np.transpose(index_map) / hm_sum #96x96


    px = np.sum(np.sum(x_score_map, axis=None))
    py = np.sum(np.sum(y_score_map, axis=None))

    return px, py

from scipy.ndimage.filters import gaussian_filter
def nms(mask):

    map = gaussian_filter(mask, sigma=4)

    map_left = np.zeros(map.shape)
    map_left[1:, :] = map[:-1, :]
    map_right = np.zeros(map.shape)
    map_right[:-1, :] = map[1:, :]
    map_up = np.zeros(map.shape)
    map_up[:, 1:] = map[:, :-1]
    map_down = np.zeros(map.shape)
    map_down[:, :-1] = map[:, 1:]

    peaks_binary = np.logical_and.reduce((map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.05))
    return peaks_binary

def extract_parts(mask):
    # Body parts location heatmap, one per part (19)
    heatmap_avg = np.zeros((96, 96, 68))
    heatmap_avg = mask
    # extract outputs, resize, and remove padding
    #heatmap_avg = cv2.resize(heatmap_avg, (96, 96), interpolation=cv2.INTER_CUBIC)
    all_peaks = []
    peak_counter = 0
    for part in range(68):
        hmap_ori = heatmap_avg[:, :, part]
        hmap = gaussian_filter(hmap_ori, sigma=3)

        # Find the pixel that has maximum value compared to those around it
        hmap_left = np.zeros(hmap.shape)
        hmap_left[1:, :] = hmap[:-1, :]
        hmap_right = np.zeros(hmap.shape)
        hmap_right[:-1, :] = hmap[1:, :]
        hmap_up = np.zeros(hmap.shape)
        hmap_up[:, 1:] = hmap[:, :-1]
        hmap_down = np.zeros(hmap.shape)
        hmap_down[:, :-1] = hmap[:, 1:]

        # reduce needed because there are > 2 arguments
        peaks_binary = np.logical_and.reduce(
            (hmap >= hmap_left, hmap >= hmap_right, hmap >= hmap_up, hmap >= hmap_down, hmap > 0.05))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        #peaks_with_score = [x + (hmap_ori[x[1], x[0]],) for x in peaks]  # add a third element to tuple with score
        idx = range(peak_counter, peak_counter + len(peaks))
        #peaks_with_score_and_id = [peaks_with_score[i] + (idx[i],) for i in range(len(idx))]

        all_peaks.append(peaks)
        peak_counter += len(peaks)

    return all_peaks

def simple(mask):
    mapSmooth = cv2.GaussianBlur(mask, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth>0.1)
    # find the blobs
    #mapMask = cv2.cvtColor(mapMask, cv2.COLOR_BGR2GRAY)
    try:
        # OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        # OpenCV3.x
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each blob find the maxima
    keypoints = []
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        #keypoints.append(maxLoc + (mask[maxLoc[1], maxLoc[0]],))
        keypoints.append(maxLoc + (mask[maxLoc[1], maxLoc[0]]))

    return mapMask, keypoints

#input_shape = (96, 96, 3)
input_shape = (96, 96, 3)
def get_model():
    return model(input_shape)
    #return model(16, 96, 96, 1, 15)  # hrnet
    #return model(input=input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)

model = get_model()

model.summary()
model.load_weights("unetcocoface2.hdf5") #multi 96x96
#model.load_weights("unetcocoface3.hdf5") #multi 96x96

#img = cv2.imread("girlphone.jpg")
img = cv2.imread("multi.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#img = cv2.resize(img, (96,96), interpolation = cv2.INTER_AREA)
img = cv2.resize(img, (96,96), interpolation = cv2.INTER_AREA)
img = np.float32(img) / 255

train_x = np.expand_dims(img, axis=0)
#train_x = np.expand_dims(img, axis=-1)
print(train_x.shape)

y_pred_train = model.predict(train_x)

# if prediction is 0, which means I am missing on the image, then show the frame in gray color.
# if y_pred_train == 0:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

plt.imshow(img)
plt.show()

plt.imshow(np.array(y_pred_train).reshape(96, 96, 68).sum(axis=-1),cmap="jet")  # 96,96
plt.show()

print(np.array(y_pred_train).reshape(96, 96, 68).sum(axis=-1).shape)  # 96,96

pred_list = []

for k in range(68):
    xpred, ypred = findCoordinates(
        np.array(y_pred_train).reshape(96, 96, 68)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list.append(xpred)
    pred_list.append(ypred)
pred_list = np.array(pred_list, dtype=np.float32)  # 42,
print(pred_list.shape)

# plot_keypoints(train_x[0], pred_list)
for i in range(0, 136, 2):
    # plt.scatter((points[i] + 0.5)*96, (points[i+1]+0.5)*96, color='red')
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
    # plt.scatter(points[:, 0], points[:, 1])
    # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
plt.imshow(img)  # gray
plt.show()

###################################################################

peaks = nms(np.array(y_pred_train).reshape(96, 96, 68))
print(peaks.shape)

masks, keypoints = simple(np.array(y_pred_train).reshape(96, 96, 68)[:, :, 30]) #96,96 #len 6 list
print(keypoints)
keypoints = np.array(keypoints) #6,3
keypoints = keypoints.flatten() #18,
print(keypoints)
print(keypoints.shape)
plt.imshow(np.array(masks).reshape(96, 96, 1),cmap="jet")  # 96,96
plt.show()
plt.imshow(np.array(masks).reshape(96, 96, 1),cmap="gray")  # 96,96
plt.show()

# plot_keypoints(train_x[0], pred_list)
for i in range(0, 10, 2):
    # plt.scatter((points[i] + 0.5)*96, (points[i+1]+0.5)*96, color='red')
    plt.scatter(keypoints[i], keypoints[i + 1], color='red')
    # plt.scatter(points[:, 0], points[:, 1])
    # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
plt.imshow(img)  # gray
plt.show()

