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

# plots keypoints on face image
def plot_keypoints(img, points):
    # display image
    plt.imshow(img, cmap='gray')
    #plt.imshow(np.float32(img), cmap='gray')
    # plot the keypoints
    for i in range(0, 136, 2):
        #plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(points[i], points[i + 1], color='red')
        # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    plt.show()

def draw_skeleton_on_image(image, keypoints, index=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    joints = []
    for i in range(0, 136, 2):
        joint_x = keypoints[i]
        joint_y = keypoints[i+1]
        joints.append((joint_x, joint_y))
    # draw skeleton
    MPII_BONES = [
        [0, 2],
        [0, 3],
        [1, 5],
        [1, 4],
        [2, 6],
        [2, 10],
        [3, 7],
        [4, 8],
        [4, 10],
        [5, 9],
        #[6, 7],
        #[8, 9],
        [10, 13],
        [11, 13],
        [11, 14],
        [12, 13],
        [12, 14]
    ]
    for bone in MPII_BONES:
        joint_1 = joints[bone[0]]
        joint_2 = joints[bone[1]]
        plt.plot([joint_1[0], joint_2[0]], [joint_1[1], joint_2[1]], linewidth=3, alpha=0.7, color="red")
    plt.axis('off')
    plt.savefig("input_skeleton.png", bbox_inches='tight', pad_inches=0)
    plt.show()

input_shape = (96, 96, 3)
def get_model():
    return model(input_shape)
    #return model(16, 96, 96, 1, 15)  # hrnet
    #return model(input=input_shape)
    #return model(input_shape=input_shape, num_classes=num_classes)

model = get_model()

model.summary()
#model.load_weights("unetcocoface.hdf5") #single 96x96
#model.load_weights("unetcocoface2.hdf5") #multi 96x96
model.load_weights("unetcocoface3.hdf5") #multi 256x256

#img = cv2.imread("girlphone.jpg")
img = cv2.imread("multi.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    # plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
    # plt.scatter(points[:, 0], points[:, 1])
    # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
plt.imshow(img)  # gray
plt.show()


