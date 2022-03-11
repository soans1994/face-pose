import pandas as pd
import cv2
import tensorflow
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical


class generator(tensorflow.keras.utils.Sequence):

    def __init__(self, directory, idxs, img_dict, labels_dict,
                 target_size=(96, 96), batch_size=32, augment=True,
                 transform_dict=None, shuffle=True):

        self.directory = directory
        self.idxs = idxs
        self.img_dict = img_dict
        self.labels_dict = labels_dict
        self.transform_dict = transform_dict
        self.target_size = target_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    # shuffle indices at the end of each epoch
    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.idxs)

    # return number of batches per epoch
    def __len__(self):

        if self.augment is True:
            multiplier = 5
        else:
            multiplier = 1

        return int(np.floor(len(self.idxs) * multiplier / self.batch_size))

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
                s = random.uniform(0.7, 1.0)
            else:
                s = 1.0

            r = random.randint(-10, 10)
            M_rot = cv2.getRotationMatrix2D(center=(img.shape[0] // 2, img.shape[1] // 2), angle=r, scale=s)
            img = cv2.warpAffine(img, M_rot, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=1)

        if self.transform_dict['Shift']:
            tx = random.randint(-20, 20)
            ty = random.randint(-20, 20)
            M_shift = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            img = cv2.warpAffine(img, M_shift, (img.shape[0], img.shape[1]),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=1)

        # transform keypoints
        c = (img.shape[0] // 2, img.shape[1] // 2)

        for i in range(0, len(keypoints) - 1, 2):

            px = keypoints[i]
            py = keypoints[i + 1]
            p = np.array([px, py, 1], dtype=int)

            # apply flip
            if self.transform_dict['Flip'] and flip:
                p[0] = c[0] - (p[0] - c[0])

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

        return img, aug_keypoints

    # load image from disk
    def _load_image(self, fn):

        img = cv2.imread(filename=os.path.join(self.directory, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.float32(img) / 255

        return img

    # apply gaussian kernel to image
    def _gaussian(self, xL, yL, sigma, H, W):

        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

    # convert original image to heatmap
    def _convertToHM(self, img, keypoints, sigma=2):

        H = img.shape[0]
        W = img.shape[1]
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

        j = 1
        for i in range(0, nKeypoints // 2):
            x = keypoints[i * 2]
            y = keypoints[1 + 2 * i]

            channel_hm = self._gaussian(x, y, sigma, H, W)
            channel_hm[channel_hm > 0.05] = j
            channel_hm[channel_hm < 0.05] = 0
            j = j + 1

            img_hm[:, :, i] = channel_hm

        # img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0] * img_hm.shape[1] * nKeypoints // 2, 1))
        img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0], img_hm.shape[1], nKeypoints // 2))#96x96x15
        #img_hm = np.array(img_hm).sum(axis=-1)#96x96
        img_hm = np.maximum.reduce([img_hm[:,:,0], img_hm[:,:,1], img_hm[:,:,2], img_hm[:,:,3], img_hm[:,:,4], img_hm[:,:,5]
                                    , img_hm[:,:,6], img_hm[:,:,7], img_hm[:,:,8], img_hm[:,:,9], img_hm[:,:,10], img_hm[:,:,11]
                                    , img_hm[:,:,12], img_hm[:,:,13], img_hm[:,:,14]])#96x96
        img_hm = np.expand_dims(img_hm, -1)#96x96x1
        from tensorflow.keras.utils import to_categorical
        img_hm = to_categorical(img_hm, 16)
        return img_hm

    # generate batches of scaled images and bounding boxes
    def _data_generation(self, idxs):

        x = []
        y = []

        for idx in idxs:
            img = self._load_image(self.img_dict[idx])
            keypoints = self.labels_dict[idx]

            if self.augment is True and self.transform_dict:
                img, keypoints = self._transform_image(img, keypoints)

            img = np.reshape(img, (96, 96, 1)) #removed this for rgb
            img_hm = self._convertToHM(img, keypoints)
            #img = np.stack((img,) * 3, axis=-1)  # added for repeat 3 channels to use rgb pretrained use

            x.append(img)
            y.append(img_hm)

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    # return indices for train batches
    def _get_train_idxs(self, idx):

        # number of batches in original train set
        N = int(np.floor(len(self.idxs) / self.batch_size))

        # idx exceeds original image indices
        if idx > N:

            # reset start idx
            if idx % N == 0:
                reset_idx = 0  # ((idx - 1) % N) + 1
            else:
                reset_idx = idx % N - 1

            start = reset_idx * self.batch_size

            # end idx
            if (reset_idx + 1) * self.batch_size > len(self.idxs):
                end = len(self.idxs)
            else:
                end = (reset_idx + 1) * self.batch_size

        # idx is within in original train set
        else:
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size

        return start, end

    # return indices for val batches
    def _get_val_idxs(self, idx):

        if (idx + 1) * self.batch_size > len(self.idxs):
            end = len(self.idxs)
        else:
            end = (idx + 1) * self.batch_size

        return idx * self.batch_size, end

    # return batch of image data and labels
    def __getitem__(self, idx):

        if self.augment is True:
            start_batch_idx, end_batch_idx = self._get_train_idxs(idx)
        else:
            start_batch_idx, end_batch_idx = self._get_val_idxs(idx)

        idxs = self.idxs[start_batch_idx:end_batch_idx]
        batch_x, batch_y = self._data_generation(idxs)
        #tf.keras.utils.to_categorical(batch_y, 15)

        return batch_x, batch_y


if __name__ == "__main__":
    data_dir = "face"
    train_dir = "train"
    train_csv = "training.csv"
    test_dir = "testl"
    test_csv = "test.csv"

    df_train = pd.read_csv(os.path.join(data_dir, train_csv))
    df_test = pd.read_csv(os.path.join(data_dir, test_csv))

    n_train = df_train['Image'].size
    n_test = df_test['Image'].size

    df_kp = df_train.iloc[:, 0:30]

    idxs = []

    img_dict = {}
    kp_dict = {}

    for i in range(n_train):

        if True in df_train.iloc[i, 0:30].isna().values:
            continue
        else:
            idxs.append(i)

            img_dict[i] = "train" + str(i) + ".png"

            # keypoints
            kp = df_kp.iloc[i].values.tolist()
            kp_dict[i] = kp
    random.shuffle(idxs)

    # subset = int(0.1*len(idxs))

    cutoff_idx = int(0.9 * len(idxs))
    train_idxs = idxs[0:cutoff_idx]
    val_idxs = idxs[cutoff_idx:len(idxs)]

    print("\n# of Training Images: {}".format(len(train_idxs)))
    print("# of Val Images: {}".format(len(val_idxs)))

    transform_dict = {"Flip": False, "Shift": False, "Scale": False, "Rotate": False}

    train_gen = generator(os.path.join(data_dir, train_dir),
                          train_idxs,
                          img_dict,
                          kp_dict,
                          transform_dict=transform_dict,
                          augment=False,
                          batch_size=16, shuffle=True)

    val_gen = generator(os.path.join(data_dir, test_dir),
                        val_idxs,
                        img_dict,
                        kp_dict,
                        augment=False,
                        batch_size=16)

    print("\n# of training batches= %d" % len(train_gen))
    print("# of validation batches= %d" % len(val_gen))

    index = 7
    train_imgs, train_masks = train_gen[index]
    print(train_imgs.shape)
    print(train_masks.shape)

    for train_x, train_y in train_gen:
        print(train_x.shape, train_y.shape)
        print(train_x[index].shape, train_y[index].shape)
        break

    plt.imshow(train_x[index])
    plt.show()
    plt.imshow(np.argmax(train_y[index],axis=-1),cmap="jet")  # for 2nd
    plt.axis('off')
    plt.savefig("categorical_mask.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    print(np.unique(train_y[index]))

    plt.imshow(np.array(train_y[index]).reshape(96, 96, 16).sum(axis=-1))  # for 2nd
    plt.show()
    plt.imshow(np.array(train_y[index]).reshape(96, 96, 16).sum(axis=-1), cmap="gray")  # for 2nd
    plt.axis('off')
    plt.show()

    plt.imshow(train_y[index].reshape(96, 96, 16)[:, :, 0],cmap="gray")  # 368,368
    plt.axis('off')
    plt.savefig("c1.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.imshow(train_y[index].reshape(96, 96, 16)[:, :, 1],cmap="gray")  # 368,368
    plt.axis('off')
    plt.savefig("c2.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.imshow(train_y[index].reshape(96, 96, 16)[:, :, 2],cmap="gray")  # 368,368
    plt.axis('off')
    plt.savefig("c3.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.imshow(train_y[index].reshape(96, 96, 16)[:, :, 3],cmap="gray")  # 368,368
    plt.axis('off')
    plt.savefig("c4.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.imshow(train_y[index].reshape(96, 96, 16)[:, :, 14],cmap="gray")  # 368,368
    plt.axis('off')
    plt.savefig("c15.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    #
    def findCoordinates(mask):

        hm_sum = np.sum(mask)  # repeats 21 times for each keypoint

        index_map = [j for i in range(96) for j in range(96)]
        index_map = np.reshape(index_map, newshape=(96, 96))

        x_score_map = mask * index_map / hm_sum  # 96x96
        y_score_map = mask * np.transpose(index_map) / hm_sum  # 96x96

        px = np.sum(np.sum(x_score_map, axis=None))
        py = np.sum(np.sum(y_score_map, axis=None))

        return px, py


    gt_list = []
    pred_list = []
    for k in range(1,16):
        xpred, ypred = findCoordinates(
            np.array(train_y[index]).reshape(96, 96, 16)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
        pred_list.append(xpred)
        pred_list.append(ypred)
    pred_list = np.array(pred_list, dtype=np.float32)
    print(pred_list.shape)

    for i in range(0, 30, 2):
        # plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
        plt.scatter(pred_list[i], pred_list[i + 1], color='red')
        # plt.scatter(points[:, 0], points[:, 1])
        # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
    plt.imshow(train_x[index], cmap='gray')  # gray
    plt.axis('off')
    plt.savefig("outkeys.png", bbox_inches='tight', pad_inches=0)
    plt.show()