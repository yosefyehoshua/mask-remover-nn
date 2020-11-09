import os
import numpy as np
from matplotlib import cm
from PIL import Image
import logging
import json
import random
from data_generator.facemasker import FaceMasker



def create_mask(celeba_img, mask_path):
    face_maker = FaceMasker(celeba_img, mask_path, show=False, model='hog')
    if not celeba_img.has_mask():  # prevents double masking
        face_maker.mask()


def create_binary_mask(img_shape, mask_img, mask_xy, threshold_value=0):

    mask_face_img = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    # Load image and convert to greyscale
    mask_img = mask_img.convert("L")

    imgData = np.asarray(mask_img)
    thresholdedData = (imgData > threshold_value) * 1

    # pay attention to x,y positions in mask & img
    mask_face_img[mask_xy[1]: mask_xy[1] + thresholdedData.shape[0], mask_xy[0]: mask_xy[0] + thresholdedData.shape[1]] = thresholdedData
    np_arr = np.uint8(mask_face_img * 255)
    np_arr[np_arr > 0] = 255
    return Image.fromarray(np_arr).convert("RGB")



def create_masked_faces(celeba_ds, masks_paths):
    r = list(range(len(celeba_ds)))
    random.shuffle(r)
    for i in range(len(r) // 2):
        idx = r[i]
        mask_path = random.choice(masks_paths)
        create_mask(celeba_ds[idx], mask_path)




