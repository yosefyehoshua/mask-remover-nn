import numpy as np
import os
import cv2
from models.face_alignment.face_aligner import FaceAlign
from models.face_parsing.predict import get_mask_from_arr
from models.image2stylegan.weight_convert import weight_converter
from models.image2stylegan.local_style_transfer import transfer
import copy
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--img_no_mask',default="models/image2stylegan/source_image/josef_no_mask3.jpeg") 
    arg('--img_wth_mask',default="models/image2stylegan/source_image/josef_blue_mask3.jpeg") 
    arg('--download_weights', default=False)
    return parser.parse_args()


def align_imgs(img_src, img_ref):
    face_aligner = FaceAlign()

    img_ref_np = face_aligner.align_to_ffhq(img_ref)
    
    img_src_np = cv2.imread(img_src)

    # set new align values
    face_aligner.fit_image(img_ref_np)
    img_src_np = face_aligner.transform(img_src_np)

    img_src_np = cv2.cvtColor(img_src_np, cv2.COLOR_BGR2RGB)
    img_ref_np = cv2.cvtColor(img_ref_np, cv2.COLOR_BGR2RGB)

    return img_src_np, img_ref_np


def download_and_convert_weights():
    weight_converter()

def get_mask(img_src_np):
    img_src_np = cv2.cvtColor(img_src_np, cv2.COLOR_BGR2RGB)
    mask_np = get_mask_from_arr(img_src_np)
    return mask_np

def remove_corona_mask(image_no_mask, img_wth_mask, download_weights, save_inputs=False):        
    if download_weights:
        download_and_convert_weights()
    
    img_src_np, img_ref_np = align_imgs(img_wth_mask, image_no_mask) # src_im2, src_im1
    
    mask_np = get_mask_from_arr(img_src_np)
    
    if save_inputs: # for debug
        cv2.imwrite('mask_np.png', mask_np)    
        cv2.imwrite('img_src_np.png', cv2.cvtColor(img_src_np, cv2.COLOR_BGR2RGB))    
        cv2.imwrite('img_ref_np.png', cv2.cvtColor(img_ref_np, cv2.COLOR_BGR2RGB))    

    
    transfer(img_ref_np, img_src_np, mask_np)

def main():
    args = parse_args()
    image_no_mask = args.img_no_mask # src_im1 
    img_wth_mask = args.img_wth_mask # src_im2
    download_weights = args.download_weights
    remove_corona_mask(image_no_mask, img_wth_mask, download_weights)        

if __name__ == "__main__":
    main()
