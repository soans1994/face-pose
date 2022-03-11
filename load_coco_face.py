"""
Process CMU Hand dataset to get cropped hand datasets.
"""
from tensorflow import keras
import json
import math
from tqdm import tqdm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from data_generator import generator2, generator3, generator3gray #generator2 single , generator3 multiple
from data_generator import generator22#, generator33  # binary mask generator22 single , generator33 multiple

NUM_KEYPOINTS = 68
from pycocotools.coco import COCO

# function iterates ofver all ocurrences of a  person and returns relevant data row by row
def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        url = img_meta['coco_url']

        yield [img_id, img_file_name, w, h, url, anns]

def convert_to_df(coco, data_set):
    images_data = []
    persons_data = []

    for img_id, img_fname, w, h, url, meta in get_meta(coco):
        images_data.append({
            'image_id': int(img_id),
            'src_set_image_id': int(img_id), # repeat id to reference after join
            'coco_url': url,
            'path': data_set + '/' + img_fname,
            'width': int(w),
            'height': int(h)
        })
        for m in meta:
            persons_data.append({
                'ann_id': m['id'],
                'image_id': m['image_id'],
                'is_crowd': m['iscrowd'],
                'face_valid': m['face_valid'],
                'face_box': m['face_box'],
                'face_kpts': m['face_kpts'],
                'bbox': m['bbox'],
                'bbox_area' : m['bbox'][2] * m['bbox'][3],
                'face_box_area': m['face_box'][2] * m['face_box'][3],
                'area': m['area'],
                'num_keypoints': m['num_keypoints'],
                'keypoints': m['keypoints'],
                'segmentation': m['segmentation']
            })

    images_df = pd.DataFrame(images_data)
    images_df.set_index('image_id', inplace=True)

    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index('image_id', inplace=True)

    return images_df, persons_df

def get_df(path_to_train_anns, path_to_val_anns):
    train_coco = COCO(path_to_train_anns) # load annotations for training set
    val_coco = COCO(path_to_val_anns) # load annotations for validation set
    images_df, persons_df = convert_to_df(train_coco, 'train2017')
    train_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    train_coco_df['source'] = 0
    train_coco_df.head()

    images_df, persons_df = convert_to_df(val_coco, 'val2017')
    val_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)
    val_coco_df['source'] = 1
    val_coco_df.head()

    return pd.concat([train_coco_df, val_coco_df], ignore_index=True)
    # ^ Dataframe containing all val and test keypoint annotations

def load_and_filter_annotations(path_to_train_anns, path_to_val_anns, subset):
    df = get_df(path_to_train_anns, path_to_val_anns)
    # apply filters here
    print(f"Unfiltered df contains {len(df)} anns")
    df = df.loc[df['is_crowd'] == 0]  # drop crowd anns
    df = df.loc[df['face_valid'] == 1]  # drop crowd anns
    KP_FILTERING_GT = 4  # Greater than x keypoints
    df = df.loc[df['num_keypoints'] > KP_FILTERING_GT]  # drop anns containing x kps
    #BBOX_MIN_SIZE = 900  # Filter out images smaller than 30x30, TODO tweak
    BBOX_MIN_SIZE = 100  # Filter out images smaller than 10x10, TODO tweak
    #df = df.loc[df['bbox_area'] > BBOX_MIN_SIZE]  # drop small bboxes
    df = df.loc[df['face_box_area'] > BBOX_MIN_SIZE]  # drop small bboxes
    train_df = df.loc[df['source'] == 0]
    val_df = df.loc[df['source'] == 1]
    if subset != 1.0:
        train_df = train_df.sample(frac=subset, random_state=1)
    print(f"Train/Val dfs contains {len(train_df)}/{len(val_df)} anns")
    return train_df.reset_index(), val_df.reset_index()

if __name__=="__main__":

    train_annot_path = 'coco/annotations/coco_wholebody_train_v1.0.json'
    val_annot_path = 'coco/annotations/coco_wholebody_val_v1.0.json'
    train_img_path = 'coco/'

    train_df, val_df = load_and_filter_annotations(train_annot_path, val_annot_path, subset=1.0)

    x = generator22(train_df, train_img_path, shuffle=True, batch_size=16, input_dim=(256,256), output_dim=(256,256)) #cropped face
    #x = generator3(train_df, train_img_path, shuffle=True, batch_size=16, input_dim=(256, 256), output_dim=(256, 256)) #resized uncropped face
    #train_images, train_labels = generator(samples, batch_size=32, aug=None)#, aug=train_aug)
    #print(len(x), len(y))
    print("length of train genearator",len(x))

    index = 0
    for train_x,train_y in x:
        print(train_x.shape, train_y.shape)
        print(train_x[index].shape, train_y[index].shape)
        break


    cv2.imshow("data", train_x[index])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_x[index])
    plt.axis('off')
    plt.savefig("cface1.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_y[index].sum(axis=-1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_y[index].sum(axis=-1))
    plt.axis('off')
    plt.savefig("cmask1.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_x[index+1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_x[index+1])
    plt.axis('off')
    plt.savefig("cface2.png", bbox_inches='tight', pad_inches=0)
    plt.show()


    cv2.imshow("data", train_y[index+1].sum(axis=-1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_y[index+1].sum(axis=-1))
    plt.axis('off')
    plt.savefig("cmask2.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_x[index+2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_x[index+2])
    plt.axis('off')
    plt.savefig("cface3.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_y[index+2].sum(axis=-1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_y[index+2].sum(axis=-1))
    plt.axis('off')
    plt.savefig("cmask3.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_x[index+3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_x[index+3])
    plt.axis('off')
    plt.savefig("cface4.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_y[index+3].sum(axis=-1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_y[index+3].sum(axis=-1))
    plt.axis('off')
    plt.savefig("cmask4.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_x[index+4])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_x[index+4])
    plt.axis('off')
    plt.savefig("cface5.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_y[index+4].sum(axis=-1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_y[index+4].sum(axis=-1))
    plt.axis('off')
    plt.savefig("cmask5.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_x[index+5])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_x[index+5])
    plt.axis('off')
    plt.savefig("cface6.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    cv2.imshow("data", train_y[index+5].sum(axis=-1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(train_y[index+5].sum(axis=-1))
    plt.axis('off')
    plt.savefig("cmask6.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.imshow(train_y[index][:,:,0])  # 368,368
    plt.show()

    cv2.imshow("data", train_y[index][:,:,0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imshow(train_y[index][:,:,16])  # 368,368
    plt.show()

    cv2.imshow("data", train_y[index][:, :, 67])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


