import numpy as np 
import matplotlib.pyplot as plt 
from models.image2stylegan.stylegan_layers import G_mapping,G_synthesis, get_noise_params, delete_noise_param
from models.image2stylegan.read_image import image_reader, np_reader
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image
from models.image2stylegan.perceptual_model import VGG16_for_Perceptual
import torch.optim as optim
import models.image2stylegan.cfg as cfg

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def transfer(src_im1, src_im2, mask):

     g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=cfg.resolution))    
    ]))

     g_all.load_state_dict(torch.load(cfg.weight_file, map_location=device))
     g_all.eval()
     g_all.to(device)
     g_mapping, g_synthesis=g_all[0],g_all[1]


     img_0=np_reader(src_im1).to(device)  # (1,3,1024,1024) 

     img_1=np_reader(src_im2).to(device)  # (1,3,1024,1024)

     blur_mask0=np_reader(mask).to(device) # (1,1,1024,1024)

     blur_mask0=blur_mask0[:,0,:,:].unsqueeze(0)
     blur_mask1=blur_mask0.clone()
     blur_mask1=1-blur_mask1

     MSE_Loss=nn.MSELoss(reduction="mean")
     upsample2d=torch.nn.Upsample(scale_factor=0.5, mode='bilinear')

     img_p0=img_0.clone()  # resize for perceptual net
     img_p0=upsample2d(img_p0)
     img_p0=upsample2d(img_p0)  # (1,3,256,256)

     img_p1=img_1.clone()
     img_p1=upsample2d(img_p1)
     img_p1=upsample2d(img_p1)  # (1,3,256,256)

     perceptual_net = VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device)
     # conv1_1, conv1_2, conv2_2, conv3_3
     dlatent = torch.zeros((1,18,512),requires_grad=True,device=device)
     optimizer=optim.Adam({dlatent},lr=0.01,betas=(0.9,0.999),eps=1e-8)

     print("Start")
     loss_list=[]
     for i in range(cfg.iteration):
          optimizer.zero_grad()
          synth_img = g_synthesis(dlatent)
          
          synth_img = (synth_img + 1.0) / 2.0
          loss_wl0=caluclate_loss(synth_img,img_0,perceptual_net,img_p0,blur_mask0,MSE_Loss,upsample2d)
          loss_wl1=caluclate_loss(synth_img,img_1,perceptual_net,img_p1,blur_mask1,MSE_Loss,upsample2d)
          loss=loss_wl0+loss_wl1
          loss.backward()

          optimizer.step()
          
          loss_np=loss.detach().cpu().numpy()
          loss_0=loss_wl0.detach().cpu().numpy()
          loss_1=loss_wl1.detach().cpu().numpy()

          loss_list.append(loss_np)
          #print(len(get_noise_params()))
          if i%100==0:
              print("latent_W iter{}: loss -- {},  loss0 --{},  loss1 --{"
                    "}".format(i,
                                                                          loss_np,
                                                                          loss_0,
                                                                          loss_1))
              save_image(synth_img.clamp(0, 1),
                         "models/image2stylegan/save_image/local_/{}.png".format(i))
              np.save("models/image2stylegan/latent_W/crossover.npy", dlatent.detach().cpu().numpy())

          if i != (cfg.iteration - 1):
              delete_noise_param()


     noise_arr = get_noise_params()
     # noise optimization
     optimizer_n = optim.Adam(noise_arr,lr=5,betas=(0.9, 0.999), eps=1e-8)

     for i in range(cfg.iteration):
          optimizer.zero_grad()
          synth_img = g_synthesis(dlatent)
          synth_img = (synth_img + 1.0) / 2.0
          loss_n = caluclate_loss_n(synth_img, img_0, img_1, blur_mask0, blur_mask1, MSE_Loss)
          loss_n.backward()
          optimizer_n.step()

          loss_np = loss_n.detach().cpu().numpy()

          if i%100==0:
               print("noise_N iter{}: loss -- {}".format(i,loss_np))
               save_image(synth_img.clamp(0,1),"models/image2stylegan/save_image_noise/crossover/{}.png".format(i))
               np.save("models/image2stylegan/noise_N/crossover.npy",dlatent.detach().cpu().numpy())
          delete_noise_param()

def caluclate_loss_n(synth_img, img_0, img_1, blur_mask0, blur_mask1, MSE_Loss):
     mse_loss_0 = MSE_Loss(synth_img * blur_mask0.expand(1,3,1024,1024),img_0 * blur_mask0.expand(1,3,1024,1024))
     mse_loss_1 = MSE_Loss(synth_img * blur_mask1.expand(1,3,1024,1024),img_1 * blur_mask1.expand(1,3,1024,1024))
     return mse_loss_0 + mse_loss_1


def caluclate_loss(synth_img,img,perceptual_net,img_p,blur_mask,MSE_Loss,upsample2d): #W_l
     #calculate MSE Loss
     mse_loss=MSE_Loss(synth_img*blur_mask.expand(1,3,1024,1024),img*blur_mask.expand(1,3,1024,1024)) # (lamda_mse/N)*||G(w)-I||^2
     #calculate Perceptual Loss
     real_0,real_1,real_2,real_3=perceptual_net(img_p)
     synth_p=upsample2d(synth_img) #(1,3,256,256)
     synth_p=upsample2d(synth_p)
     synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)

     perceptual_loss=0
     blur_mask=upsample2d(blur_mask)
     blur_mask=upsample2d(blur_mask) #(256,256)

     perceptual_loss+=MSE_Loss(synth_0*blur_mask.expand(1,64,256,256),real_0*blur_mask.expand(1,64,256,256))
     perceptual_loss+=MSE_Loss(synth_1*blur_mask.expand(1,64,256,256),real_1*blur_mask.expand(1,64,256,256))
     blur_mask=upsample2d(blur_mask) 
     blur_mask=upsample2d(blur_mask) #(64,64)
     perceptual_loss+=MSE_Loss(synth_2*blur_mask.expand(1,256,64,64),real_2*blur_mask.expand(1,256,64,64))
     blur_mask=upsample2d(blur_mask) #(64,64)
     perceptual_loss+=MSE_Loss(synth_3*blur_mask.expand(1,512,32,32),real_3*blur_mask.expand(1,512,32,32))

     return mse_loss+perceptual_loss


if __name__ == "__main__":
    transfer()



