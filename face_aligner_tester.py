import cv2
import numpy as np
import imageio
from models.face_alignment.face_aligner import FaceAlign
from models.face_alignment.landmarks_detector import LandmarksDetector
import bz2
from PIL import Image

root = '/home/josefy/mask-remover-nn/models/image2stylegan/source_image/'
img_src_path = 'almog_no_mask_white_bkg_6.jpeg'
img_dest_path = 'almog_mask_white_bkg_6.jpeg'
img_jos_path = 'josef_no_mask_profile.jpeg'
ffhq = '0.png'

ffhq_img = cv2.imread(root + ffhq)
img_mask = cv2.imread(root + img_dest_path)
img_no_mask = cv2.imread(root + img_src_path)
jos_no_mask = cv2.imread(root + img_jos_path)


face_aligner = FaceAlign()
face_aligner.fit_to_values()

#img_tr_mask = face_aligner.transform(img_mask)
img_tr_mask = face_aligner.transform(img_mask)

img_no_mask = cv2.cvtColor(img_no_mask, cv2.COLOR_BGR2RGB)
img_no_mask = Image.fromarray(img_no_mask)

jos_no_mask = cv2.cvtColor(jos_no_mask, cv2.COLOR_BGR2RGB)
jos_no_mask = np.asarray(jos_no_mask)
# jos_no_mask = Image.fromarray(jos_no_mask)

img_tr_no_mask = face_aligner.align_to_ffhq(root + img_dest_path)
jos_tr_no_mask = face_aligner.align_to_ffhq(root + img_jos_path)


img_tr_mask = cv2.cvtColor(img_tr_mask, cv2.COLOR_BGR2RGB)
img_tr_no_mask = cv2.cvtColor(img_tr_no_mask, cv2.COLOR_BGR2RGB)
jos_tr_no_mask = cv2.cvtColor(jos_tr_no_mask, cv2.COLOR_BGR2RGB)