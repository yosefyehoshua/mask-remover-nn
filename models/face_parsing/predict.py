#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.face_parsing.logger import setup_logger
from models.face_parsing.model import BiSeNet

import torch

import os
import argparse
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def vis_parsing_maps(im, parsing_anno, stride, save_im=False,
                     save_path='/home/josefy/mask-remover-nn/models/face_parsing/res/test_res/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [0, 255, 170],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [255, 170, 0],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    parts_names = ['all_skin', 'skin', 'mask', 'background']
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride,
                                  interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4,
                             vis_parsing_anno_color, 0.6, 0)

    ids = np.unique(vis_parsing_anno_color)
    all_masks = (vis_parsing_anno_color[np.newaxis] == ids[:, np.newaxis,
                                                       np.newaxis,
                                                       np.newaxis])
    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        image_name_split = os.path.splitext(save_path)
        for i, mask in enumerate(all_masks):
            # print(mask.shape)
            mask = np.sum(mask, axis=2)
            mask = mask * 255
            mask = mask.astype('float32')
            mask = cv2.resize(mask, dsize=(512, 512),
                              interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(image_name_split[0] + '_' + parts_names[i] +
                        '.png',
                        mask,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    mask = all_masks[2]
    mask = np.sum(mask, axis=2)
    mask = mask * 255
    mask = mask.astype('float32')
    mask = cv2.resize(mask, dsize=(1024, 1024),
                      interpolation=cv2.INTER_CUBIC)

    # returns mask of 'corona mask'
    return mask

def init_net():
    n_classes = 3
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = '/home/josefy/mask-remover-nn/models/face_parsing/res/cp' \
               '/16249_iter.pth'
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    return net


def test(dspth, respth='/home/josefy/mask-remover-nn/models/face_parsing'
                          '/res'
                          '/test_res'):
    if not os.path.exists(respth):
        os.makedirs(respth)

    net = init_net()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            return vis_parsing_maps(image, parsing, stride=1, save_im=True,
                                    save_path=osp.join(respth, image_path))
def get_mask_from_arr(img_np, save_im=False):
    net = init_net()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = Image.fromarray(img_np, 'RGB')

    with torch.no_grad():
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        return vis_parsing_maps(image, parsing, stride=1, save_im=save_im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--dspth', type=str, help='images to segmentation path')

    args = parser.parse_args()
    imgs_path = args.dspth

    test(dspth=imgs_path)
