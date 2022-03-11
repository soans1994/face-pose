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

#random.shuffle(idxs)#remove for same index data

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
                              batch_size=16,shuffle=True)

validation_generator = generator(os.path.join(data_dir, train_dir),
                            val_idxs,
                            img_dict,
                            kp_dict,
                            augment=False,
                            batch_size=16,shuffle=False)

print(len(train_generator), len(validation_generator))
input_shape = (96, 96, 1)
#input_shape = (256, 256, 3)
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
#model.load_weights("test.hdf5")
#model.load_weights("unetfaceccelcategoricalmask.hdf5")
model.load_weights("unetcbamfaceccelcategoricalmask.hdf5")#best categorical multi
#model.load_weights("unetcbam2faceccelcategoricalmask.hdf5")#no?
#model.load_weights("unetsenet2faceccelcategoricalmask.hdf5")
#model.load_weights("unetecanet2faceccelcategoricalmask.hdf5")

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
    break
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
    break
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
    break
#iou_total = np.array(iou_total)#array is [1, 2, 3,....]
#dice_total = np.array(dice_total)
print("val classwise iou is", iou_total / len(validation_generator))
print("val classwise dice is", dice_total / len(validation_generator))
print("K.mean iou is", K.mean(iou_total / len(validation_generator)))
print("K.mean dice is", K.mean(dice_total / len(validation_generator)))


#for testing validation data
#train_x = val_x
#train_y = val_y

plt.imshow(train_x[index])  # 368,368 for 368,368,21
plt.axis('off')
plt.savefig("input_image.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.argmax(train_y[index],axis=-1))  # 96x96
plt.axis('off')
plt.savefig("input_heatmap.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.argmax(train_y[index],axis=-1), cmap="jet")  # 96x96
plt.axis('off')
plt.savefig("input_heatmap2.png", bbox_inches='tight', pad_inches=0)
plt.show()
#plt.imshow(np.array(train_y[0]).sum(axis=-1))  # 96x96
#plt.show()
#plt.imshow(np.array(train_y[0]).sum(axis=-1), cmap="gray")  # 96x96
#plt.show()
#plt.imshow(train_y[0][:,:,0],cmap="gray")  # 368,368,21 to 368,368
#plt.show()
#plt.imshow(train_y[0][:, :, 20])  # 368,368,21 to 368,368
#plt.show()

def findCoordinates(mask):

    hm_sum = np.sum(mask) #repeats 21 times for each keypoint

    index_map = [j for i in range(96) for j in range(96)]
    index_map = np.reshape(index_map, newshape=(96,96))

    x_score_map = mask * index_map / hm_sum #96x96
    y_score_map = mask * np.transpose(index_map) / hm_sum #96x96


    px = np.sum(np.sum(x_score_map, axis=None))
    py = np.sum(np.sum(y_score_map, axis=None))

    return px, py

gt_list = []
pred_list = []

for k in range(1,16):
    xpred, ypred = findCoordinates(np.array(train_y[index]).reshape(96, 96, 16)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list.append(xpred)
    pred_list.append(ypred)
pred_list = np.array(pred_list, dtype=np.float32)
print(pred_list.shape)

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

def draw_skeleton_on_image(image, keypoints, index=None):
    fig,ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    joints = []
    for i in range(0, 30, 2):
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

#plot_keypoints(train_x[0], pred_list)
for i in range(0, 30, 2):
    # plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
    # plt.scatter(points[:, 0], points[:, 1])
    # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("input_keypoints.png", bbox_inches='tight', pad_inches=0)
plt.show()
draw_skeleton_on_image(train_x[index], pred_list)

# y_pred = model.predict(train_x)#
# print(y_pred.shape)#batch, 96x96x21, 1
# iou = mean_iou(train_y, y_pred)
# dice = mean_dice(train_y, y_pred)
# print("mean iou is", iou)
# print("mean dice is", dice)

plt.imshow(np.argmax(y_pred_train[index],axis=-1))  # 96,96
plt.axis('off')
plt.savefig("output_heatmap.png", bbox_inches='tight', pad_inches=0)
plt.show()
plt.imshow(np.argmax(y_pred_train[index],axis=-1),cmap="jet")  # 96,96
plt.axis('off')
plt.savefig("output_heatmap2.png", bbox_inches='tight', pad_inches=0)
print(np.argmax(y_pred_train[index],axis=-1).shape)#96,96
plt.show()
plt.imshow(np.argmax(y_pred_train[index],axis=-1), cmap="gray")  # 96x96
plt.show()


#plt.imshow(y_pred[0][:,:,0])  # 368,368,21 to 368,368
#plt.show()
#plt.imshow(y_pred[0][:, :, 20])  # 368,368,21 to 368,368
#plt.show()

gt_list = []
pred_list = []

# for i in range(15):
#     # calculate moments of binary image
#     M = cv2.moments(y_pred[index][:, :, i])
#     # calculate x,y coordinate of center
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     pred_list.append(cX)
#     pred_list.append(cY)
# pred_list = np.array(pred_list, dtype=np.float32)  # 42,
# print(pred_list.shape)
#
# for i in range(15):
#     # calculate moments of binary image
#     M = cv2.moments(train_y[index][:, :, i])
#     # calculate x,y coordinate of center
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     gt_list.append(cX)
#     gt_list.append(cY)
# gt_list = np.array(gt_list, dtype=np.float32)  # 42,
# print(gt_list.shape)

for k in range(1,16):
    xpred, ypred = findCoordinates(np.array(y_pred_train[index]).reshape(96, 96, 16)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    pred_list.append(xpred)
    pred_list.append(ypred)
pred_list = np.array(pred_list, dtype=np.float32) #42,
print(pred_list.shape)

for k in range(1,16):
    xtrain, ytrain = findCoordinates(np.array(train_y[index]).reshape(96, 96, 16)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
    gt_list.append(xpred)
    gt_list.append(ypred)
gt_list = np.array(gt_list, dtype=np.float32) #30,
print(gt_list.shape)

#plot_keypoints(train_x[0], pred_list)
for i in range(0, 30, 2):
    # plt.scatter((points[i] + 0.5)*256, (points[i+1]+0.5)*256, color='red')
    plt.scatter(pred_list[i], pred_list[i + 1], color='red')
    # plt.scatter(points[:, 0], points[:, 1])
    # cv2.circle(img, (int(points[i]), int(points[i + 1])), 3, (0, 255, 0), thickness=-1)  # , lineType=-1)#, shift=0)
plt.imshow(train_x[index], cmap='gray')#gray
#plt.imshow(train_x[0])#color
plt.axis('off')
plt.savefig("output_keypoints.png", bbox_inches='tight', pad_inches=0)
plt.show()

def calcRMSError(kps_gt, kps_preds):

    N = kps_gt.shape[0] * (kps_gt.shape[-1] // 2)
    error = np.sqrt(np.sum((kps_gt-kps_preds)**2)/N)

    return error
rms_error = calcRMSError(gt_list, pred_list)#42,
#rms_error = calcRMSError(gt_list, gt_list)#42,
print("Validation RMS Error = {}".format(rms_error))

#ms_error2 = mean_squared_error2(train_y[index], y_pred[index])
#ms_error = mean_squared_error(train_y[index], y_pred[index])
ms_error = mean_squared_error(gt_list, pred_list)
print("Validation MS Error = {}".format(ms_error))
#a = train_y[0].sum(axis=-1)
#b = y_pred[0].sum(axis=-1)
#c = y_pred[0][:,:,0]

# l2_dist_mean, l2_dist_std = l2_metric(kps1=gt_list, kps2=pred_list)
# print("gtlist is", gt_list)
# print("predlist is", pred_list)
# print("Validation L2 distance mean = ",l2_dist_mean)
# print("Validation L2 distance std = ",l2_dist_std)
# d = 0

print("gtlist is", gt_list)
print("predlist is", pred_list)
gt_list = np.reshape(gt_list, (15,2))
pred_list = np.reshape(pred_list, (15,2))
l2_dist_mean, l2_dist_std = l2_metric(kps1=gt_list, kps2=pred_list)# float: Mean pointwise l2 distance.
                                                                #float: Standart deviation of those distances.
print("Validation L2 distance mean = ",l2_dist_mean)
print("Validation L2 distance std = ",l2_dist_std)

gt_list = np.reshape(gt_list, (15,2))
pred_list = np.reshape(pred_list, (15,2))
print(gt_list.shape, pred_list.shape)
pck = pck_metric(kps1=gt_list, kps2=pred_list, threshold=0.5)# 0.8 best   percentage of keypoint pairs within the threshold
                                                            #Distance between predicted and true joint < 0.5
print("Percentage of Correct keypoints = ",pck)#


pck_val_total1= []
pck_val_total2= []
pck_val_total3= []
pck_val_total4= []
pck_val_total5= []
pck_val_total6= []
pck_val_total7= []
pck_val_total8= []
pck_val_total9= []
pck_val_total10= []
pck_val_total11= []
pck_val_total12= []
pck_val_total13= []
pck_val_total14= []
pck_val_total15= []
pck_val_total16= []
pck_val_total17= []
pck_val_total18= []
pck_val_total19= []
pck_val_total20= []
for val_x, val_y in tqdm(validation_generator):
    #print(train_x.shape, train_y.shape)#(16, 96, 96, 1) (16, 96, 96, 15)
    #print(train_x[index].shape, train_y[index].shape)#(96, 96, 1) (96, 96, 15)
    y_pred_val = model.predict(val_x)  #
    #print(y_pred_val.shape)  # (16, 96, 96, 15)
    pck_val1 = []
    pck_val2 = []
    pck_val3 = []
    pck_val4 = []
    pck_val5 = []
    pck_val6 = []
    pck_val7 = []
    pck_val8 = []
    pck_val9 = []
    pck_val10 = []
    pck_val11 = []
    pck_val12 = []
    pck_val13 = []
    pck_val14 = []
    pck_val15 = []
    pck_val16 = []
    pck_val17 = []
    pck_val18 = []
    pck_val19 = []
    pck_val20 = []
    for index in range(16):
        gt_list = []
        for k in range(15):
            xpred, ypred = findCoordinates(
                np.array(val_y[index]).reshape(96, 96, 16)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
            gt_list.append(xpred)
            gt_list.append(ypred)
        gt_list = np.array(gt_list, dtype=np.float32)  # 42,
        # print(pred_list.shape)
        gt_list = np.reshape(gt_list, (15, 2))

        pred_list = []
        for k in range(15):
            xpred, ypred = findCoordinates(
                np.array(y_pred_val[index]).reshape(96, 96, 16)[:, :, k])  # maskToKeypoints(mask_pred[:, :, k])
            pred_list.append(xpred)
            pred_list.append(ypred)
        pred_list = np.array(pred_list, dtype=np.float32)  # 42,
        # print(pred_list.shape)
        pred_list = np.reshape(pred_list, (15, 2))
        pck1 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=0.2) #euclidian distance is l2 norm 0.8 best
        pck2 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=0.4)
        pck3 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=0.6)
        pck4 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=0.8)
        pck5 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=1.0)
        pck6 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=1.2)
        pck7 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=1.4)
        pck8 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=1.6)
        pck9 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=1.8)
        pck10 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=2.0)
        pck11 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=2.2)
        pck12 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=2.4)
        pck13 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=2.6)
        pck14 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=2.8)
        pck15 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=3.0)
        pck16 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=3.2)
        pck17 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=3.4)
        pck18 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=3.6)
        pck19 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=3.8)
        pck20 = pck_metric(kps1=gt_list, kps2=pred_list, threshold=4.0)
        pck_val1.append(pck1)
        pck_val2.append(pck2)
        pck_val3.append(pck3)
        pck_val4.append(pck4)
        pck_val5.append(pck5)
        pck_val6.append(pck6)
        pck_val7.append(pck7)
        pck_val8.append(pck8)
        pck_val9.append(pck9)
        pck_val10.append(pck10)
        pck_val11.append(pck11)
        pck_val12.append(pck12)
        pck_val13.append(pck13)
        pck_val14.append(pck14)
        pck_val15.append(pck15)
        pck_val16.append(pck16)
        pck_val17.append(pck17)
        pck_val18.append(pck18)
        pck_val19.append(pck19)
        pck_val20.append(pck20)
    #print("Percentage of Correct keypoints single batch= ", sum(pck_val) / 16)  #
    pck_val_total1.append(sum(pck_val1) / 16)
    pck_val_total2.append(sum(pck_val2) / 16)
    pck_val_total3.append(sum(pck_val3) / 16)
    pck_val_total4.append(sum(pck_val4) / 16)
    pck_val_total5.append(sum(pck_val5) / 16)
    pck_val_total6.append(sum(pck_val6) / 16)
    pck_val_total7.append(sum(pck_val7) / 16)
    pck_val_total8.append(sum(pck_val8) / 16)
    pck_val_total9.append(sum(pck_val9) / 16)
    pck_val_total10.append(sum(pck_val10) / 16)
    pck_val_total11.append(sum(pck_val11) / 16)
    pck_val_total12.append(sum(pck_val12) / 16)
    pck_val_total13.append(sum(pck_val13) / 16)
    pck_val_total14.append(sum(pck_val14) / 16)
    pck_val_total15.append(sum(pck_val15) / 16)
    pck_val_total16.append(sum(pck_val16) / 16)
    pck_val_total17.append(sum(pck_val17) / 16)
    pck_val_total18.append(sum(pck_val18) / 16)
    pck_val_total19.append(sum(pck_val19) / 16)
    pck_val_total20.append(sum(pck_val20) / 16)
    break
print("Percentage of Correct keypoints all batches with threshold 0.8= ", sum(pck_val_total4) / len(validation_generator))
print(sum(pck_val_total1) / len(validation_generator),  sum(pck_val_total2) / len(validation_generator),  sum(pck_val_total3) / len(validation_generator),  sum(pck_val_total4) / len(validation_generator),  sum(pck_val_total5) / len(validation_generator),  sum(pck_val_total6) / len(validation_generator),  sum(pck_val_total7) / len(validation_generator),  sum(pck_val_total8) / len(validation_generator),  sum(pck_val_total9) / len(validation_generator),  sum(pck_val_total10) / len(validation_generator),  sum(pck_val_total11) / len(validation_generator),  sum(pck_val_total12) / len(validation_generator),  sum(pck_val_total13) / len(validation_generator),  sum(pck_val_total14) / len(validation_generator),  sum(pck_val_total15) / len(validation_generator),  sum(pck_val_total16) / len(validation_generator),  sum(pck_val_total17) / len(validation_generator),  sum(pck_val_total18) / len(validation_generator),  sum(pck_val_total19) / len(validation_generator),  sum(pck_val_total20) / len(validation_generator))
