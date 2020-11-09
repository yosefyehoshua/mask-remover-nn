#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.face_parsing.logger import setup_logger
from models.face_parsing.model import BiSeNet
from models.face_parsing.face_dataset import FaceMask
from models.face_parsing.loss import OhemCELoss
from models.face_parsing.evaluate import evaluate
from models.face_parsing.optimizer import Optimizer
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import glob
import os.path as osp
import logging
import time
import datetime
import argparse

respth = './res'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--local_rank', dest='local_rank', type=int, default=0)
    arg('--data_json', type=str,
        default='/home/josefy/mask-remover-nn/data_generator'
                '/processed_data_json')
    return parser.parse_args()


def train():
    args = parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:33241',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    setup_logger(respth)
    # dataset
    n_classes = 3
    n_img_per_gpu = 16
    n_workers = 4  # 8
    cropsize = [448, 448]

    data_root = args.data_json
    list_of_files = glob.glob(data_root + '/*.json')
    ds_json = max(list_of_files, key=os.path.getctime)
    input_file = open(ds_json)
    json_arr = json.load(input_file)

    json_arr = json_arr[:10000]  # for debug

    split_idx = int(len(json_arr) * 0.8)
    json_train = json_arr[:split_idx]
    json_val = json_arr[split_idx:]

    ds_train = FaceMask(json_train, cropsize=cropsize, mode='train')
    ds_val = FaceMask(json_val, cropsize=cropsize, mode='val')



    sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    dl_train = DataLoader(ds_train,
                    batch_size=n_img_per_gpu,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    dl_val = DataLoader(ds_val,
                          batch_size=4,
                          shuffle=False,
                          num_workers=n_workers,
                          pin_memory=True,
                          drop_last=True)

    # model
    ignore_idx = -100
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank, ],
                                              output_device=args.local_rank
                                              )
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 26000  # 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
        model=net.module,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl_train)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl_train)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = LossP(out, lb)
        loss2 = Loss2(out16, lb)
        loss3 = Loss3(out32, lb)
        loss = lossp + loss2 + loss3
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        #  print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                time=t_intv,
                eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed
        if dist.get_rank() == 0:
            if (it + 1) % int(max_iter * 0.0625) == 0:
                state = net.module.state_dict() if hasattr(net,
                                                           'module') else net.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state, './res/cp/{}_iter.pth'.format(it))

                mIOU = evaluate(dl_val, cp='cp/{}_iter.pth'.format(it))
                logger.info('mIOU is: {:.6f} | in iter: {}'.format(mIOU, it))

    #  dump the final model
    save_pth = osp.join(respth, 'model_final_diss.pth')
    # net.cpu()
    state = net.module.state_dict() if hasattr(net,
                                               'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    train()
