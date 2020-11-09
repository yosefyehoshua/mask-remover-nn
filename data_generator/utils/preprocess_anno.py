import os.path as osp
import os
import cv2
from PIL import Image
import numpy as np


def merge_masks(celeba_ds):

    counter = 0
    for img_ds in celeba_ds:

        atts = ['skin', 'face_mask']
        mask = np.zeros((512, 512))
        for l, att in enumerate(atts, 1):  # kept this loop is in case of
            # multiple merge atts
            if att in img_ds.gt_dict.keys():

                path = img_ds.gt_dict[att]

                if os.path.exists(path):
                    sep_mask = np.array(Image.open(path).convert('P'))

                    mask[sep_mask == 225] = l

        merged_pth = '{}/{}.png'.format(img_ds.gt_dir, img_ds.img_index)
        cv2.imwrite(merged_pth, mask)
        print('merged: {}, unique vals: {}'.format(merged_pth, np.unique(
            mask)))
        img_ds.merged_gt = merged_pth
        counter += 1
    print('merged {} images'.format(counter))

