import os
import sys
import argparse
from data_generator.facemasker_helper import create_masked_faces
from data_generator.utils.file_manager import write_json_arr
from data_generator.utils.preprocess_anno import merge_masks
from data_generator.processed_data import create_celeba_arr
import time
import json

def gen_data(root_dir, masks_arr):
    img_dir = root_dir + '/CelebA-HQ-img'
    anno_dir = root_dir + '/CelebAMask-HQ-mask-anno'

    celeba_ds = create_celeba_arr(img_dir, anno_dir)

    # paste masks to dataset
    print("create masks")
    create_masked_faces(celeba_ds, masks_arr)

    # merge all ground trues
    print("merge masks")
    merge_masks(celeba_ds)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    # write data json
    write_json_arr(celeba_ds,
                   '/home/josefy/mask-remover-nn/data_generator/processed_data_json/' + timestr + '.json')

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--dataset_root', type=str, help='dataset root path',
        default='/home/josefy/mask-remover-nn/CelebAMask-HQ')
    arg('--masks_path', type=str, help='corona masks vector images', default='/home/josefy/mask-remover-nn/data_generator/face_masks_images')

    args = parser.parse_args()
    masks_path = args.masks_path
    root_dir = args.dataset_root

    masks_arr = [os.path.join(masks_path, f) for f in
                    os.listdir(masks_path) if
                    os.path.isfile(os.path.join(masks_path, f))]

    gen_data(root_dir, masks_arr)




if __name__ == "__main__":
    main()
