"""
Process CMU Hand dataset to get cropped hand datasets.
"""
from tensorflow import keras
import json
import math
import random
from tqdm import tqdm
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import os

IMG_SIZE = 96#368
NUM_KEYPOINTS = 42

class generator(keras.utils.Sequence):
    def __init__(self, image_keys, aug, batch_size, target_size=(368,368), transform_dict = None, train=True):
        self.image_keys = image_keys
        self.aug = aug
        self.transform_dict = transform_dict
        self.batch_size = batch_size
        self.target_size = target_size
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_keys) // self.batch_size #'Denotes the number of batches per epoch'(samples/batch size)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_keys))#'Updates indexes after each epoch'
        if self.train:
            np.random.shuffle(self.indexes)

    # check if transformed point is located within image boundaries
    def _checkBoundaries(self, p):

        # x dimension
        if p[0] < 0:
            px = 0
        elif p[0] > self.target_size[0]:
            px = self.target_size[0]
        else:
            px = p[0]

        # y dimension
        if p[1] < 0:
            py = 0
        elif p[1] > self.target_size[1]:
            py = self.target_size[1]
        else:
            py = p[1]

        return (int(px), int(py))

    # apply shifts, rotations, scaling and flips to original image and keypoints
    def _transform_image(self, img, keypoints):

        aug_keypoints = []
        c = (img.shape[0] // 2, img.shape[1] // 2)

        if self.transform_dict['Flip']:
            flip = random.choice([True, False])
            if flip:
                img = cv2.flip(img, flipCode=1)

        if self.transform_dict['Rotate']:

            if self.transform_dict['Scale']:
                s = random.uniform(0.8, 1.2)
            else:
                s = 1.0

            r = random.randint(-20, 20)
            M_rot = cv2.getRotationMatrix2D(center=(img.shape[0] // 2, img.shape[1] // 2), angle=r, scale=s)
            img = cv2.warpAffine(img, M_rot, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=1)

        if self.transform_dict['Shift']:
            tx = random.randint(-10, 10)
            ty = random.randint(-10, 10)
            M_shift = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            img = cv2.warpAffine(img, M_shift, (img.shape[0], img.shape[1]),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=1)

        # transform keypoints
        c = (img.shape[0] // 2, img.shape[1] // 2)
        #print("lenth of keypoints is",len(keypoints))
        for i in range(0, len(keypoints) - 1, 2):

            px = keypoints[i]
            py = keypoints[i + 1]
            p = np.array([px, py, 1], dtype=int)

            # apply flip
            if self.transform_dict['Flip'] and flip:
                p[0] = c[0] - (p[0] - c[0])
                #print(p.shape)
            # apply rotation
            if self.transform_dict['Rotate']:
                p = np.dot(M_rot, p)

            # apply horizontal / vertical shifts
            if self.transform_dict['Shift']:
                p[0] += tx
                p[1] += ty

            p = self._checkBoundaries(p)

            aug_keypoints.append(p[0])
            aug_keypoints.append(p[1])
        #print("point")
        aug_keypoints = np.array(aug_keypoints)
        #print(img.shape,aug_keypoints.shape)
        return img, aug_keypoints

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        image_keys_temp = [self.image_keys[k] for k in indexes]
        images, heatmaps = self.__data_generation(image_keys_temp)
        return images, [heatmaps, heatmaps, heatmaps, heatmaps, heatmaps]

    # Function to create heatmaps by convoluting a 2D gaussian kernel over a (x,y) keypoint.
    def gaussian(self, xL, yL, H, W, sigma=5):

        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))#21,368,368
        return channel

    def __data_generation(self, image_keys_temp):
        batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
        batch_heatmaps = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 21), dtype="float32")#for 328,328,21
        #batch_heatmaps = np.empty((self.batch_size, 96*96*21, 1), dtype="float32")

        for i, key in enumerate(image_keys_temp):
            data = cv2.imread(key)
            data = cv2.resize(data, (96, 96))
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            #data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            data = np.float32(data) / 255  # make sure batch images is dtype="float32")
            data = np.reshape(data, newshape=(96, 96, 3))  # for gray
            #split = os.path.split(key)
            #print(split)
            extension = os.path.splitext(key)[0]
            #print(extension)
            key2 = extension + ".json"
            #print(key2)
            #data = cv2.resize(data, (368, 368))
            # We then project the original image and its keypoint coordinates.
            #current_image = data
            # Apply the augmentation pipeline.
            #new_image = self.aug(image=current_image)
            #new_image = current_image
            #batch_images[i,] = new_image

            dat = json.load(open(key2))
            pts = np.array(dat['hand_pts'])
            pts = pts[:, :2]  # shape 21,2
            # pts = np.array(pts).reshape(-1, 42)#1,42
            #pts = pts.flatten()  # len 42, shape 42,
            Rx = 96 / 368
            Ry = 96 / 368
            #pts = round(Rx * pts)
            pts = Rx * pts
            # current_keypoint = np.array(data["joints"])[:, :2]
            # kps = []
            pts = pts.flatten()  # len 42, shape 42,

            if self.aug is True and self.transform_dict:
                data, pts = self._transform_image(data, pts)
            data = np.reshape(data, newshape=(96, 96, 3))  # for gray
            # More on why this reshaping later.
            # batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 42)#same as below
            heatmaps =[]
            for i2 in range(0, 42, 2):
                x = int(pts[i2])
                y = int(pts[i2 + 1])
                heatmap = self.gaussian(x, y, 96, 96)#21,368,368
                heatmaps.append(heatmap)
            heatmaps = np.array(heatmaps)#21,368,368
            #heatmaps = heatmaps.sum(axis=0)#368,368
            heatmaps = np.transpose(heatmaps, axes=(1, 2, 0))  # 368,368,21  need this for conv output to match
            #heatmaps = np.reshape(heatmaps, newshape=(96 * 96 * 42 // 2, 1))
            batch_images[i,] = np.array(data, dtype=np.float32)
            batch_heatmaps[i,] = np.array(heatmaps, dtype=np.float32)
            # Scale the coordinates to [0, 1] range.
        return batch_images, batch_heatmaps

if __name__=="__main__":
    #samples = sorted(glob.glob("hand_labels_synth/synth4/*.jpg"))
    #samples = sorted(glob.glob("hand_labels_synth/synth4val/*.jpg"))
    #samples = sorted(glob.glob("hand_labels_synth/synth232/*.jpg"))
    samples = sorted(glob.glob("hand_labels_synth/synthsmall/*.jpg"))
    transform_dict = {"Flip": True, "Shift": False, "Scale": False, "Rotate": False}  # scale and rotate together
    x = generator(samples, batch_size=8, aug=True, transform_dict=transform_dict)  # , aug=train_aug)
    #train_images, train_labels = generator(samples, batch_size=32, aug=None)#, aug=train_aug)
    #print(len(x), len(y))
    #print(len(x))
    index = 4
    stage = 4
    for train_x,train_y in x:
        print(train_x.shape)
        print(train_x[stage][index].shape, train_y[index].shape)
        break
    plt.imshow(train_x[index])  # 368,368 for 368,368,21
    plt.show()
    plt.imshow(train_y[stage][index].sum(axis=-1))  # 368,368 for 368,368,21
    plt.show()
    plt.imshow(train_x[index+1])  # 368,368 for 368,368,21
    plt.show()
    plt.imshow(train_y[stage][index+1].sum(axis=-1))  # 368,368 for 368,368,21
    plt.show()


    def findCoordinates(mask):

        hm_sum = np.sum(mask)  # repeats 21 times for each keypoint

        index_map = [j for i in range(96) for j in range(96)]
        index_map = np.reshape(index_map, newshape=(96, 96))

        x_score_map = mask * index_map / hm_sum  # 96x96
        y_score_map = mask * np.transpose(index_map) / hm_sum  # 96x96

        px = np.sum(np.sum(x_score_map, axis=None))
        py = np.sum(np.sum(y_score_map, axis=None))

        return px, py


    pred_list = []
    for k in range(21):
        xpred, ypred = findCoordinates(np.array(train_y[0][index]).reshape(96, 96, 21)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
        pred_list.append(xpred)
        pred_list.append(ypred)
    pred_list = np.array(pred_list, dtype=np.float32) #42,
    print(pred_list.shape)

    for i in range(0, 42, 2):
        # plt.scatter((points[i] + 0.5)*96, (points[i+1]+0.5)*96, color='red')
        plt.scatter(pred_list[i], pred_list[i + 1], color='red')
        # plt.scatter(points[:, 0], points[:, 1])
        # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    plt.imshow(train_x[index])#gray
    #plt.imshow(train_x[0])#color
    plt.axis('off')
    plt.savefig("output_keypoints.png", bbox_inches='tight', pad_inches=0)
    plt.show()
