import os
import cv2
import warnings
import numpy as np
from .utils import read_image
from skimage.transform import warp, AffineTransform, resize
import scipy.ndimage
import PIL.Image
import sys
import bz2
from tensorflow.keras.utils import get_file

#from keras.utils import get_file
from models.face_alignment.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))


landmarks_detector = LandmarksDetector(landmarks_model_path)

eye_cascade = cv2.CascadeClassifier(
    "models/face_alignment/haarcascades"
    "/haarcascade_eye.xml")


class FaceAlign(object):

    def fit_image(self, target_image):
        self.target_width_ = target_image.shape[1]
        self.target_height_ = target_image.shape[0]
        eyes = eye_cascade.detectMultiScale(target_image, 1.1, 4)
        props = self._calc_eye_properties(eyes)
        self.eyes_mid_point_ = props[0]
        self.eye_distance_ = props[1]
        return self



    def fit_to_values(self, target_width=1024, target_height=1024,
                   eyes_mid_point=np.array([499., 485.]), eye_distance=252.03174403237384):
        # default values from Flicker-Faces HQ-dataset AVG front image
        # alignment
        self.target_width_ = target_width
        self.target_height_ = target_height
        self.eyes_mid_point_ = eyes_mid_point
        self.eye_distance_ = eye_distance
        return self

    def _calc_eye_properties(self, eyes):
        left_eye, right_eye = self._get_eyes(eyes)
        eyes_mid_point = (left_eye + right_eye)/2.0
        eye_distance = np.sqrt(np.sum(np.square(left_eye - right_eye)))

        return eyes_mid_point, eye_distance

    def _get_eyes(self, eyes):
        # Calculating coordinates of a central points of the rectangles
        left_eye_x = int(eyes[0][0] + eyes[0][2] / 2)
        left_eye_y = int(eyes[0][1] + eyes[0][3] / 2)
        right_eye_x = int(eyes[1][0] + eyes[1][2] / 2)
        right_eye_y = int(eyes[1][1] + eyes[1][3] / 2)

        return np.array([left_eye_x, left_eye_y]), np.array([right_eye_x,
                                                            right_eye_y])


    def transform(self, img):

        img_eyes = eye_cascade.detectMultiScale(img, 1.1, 4)

        eyes_mid_point, eye_distance = self._calc_eye_properties(img_eyes)

        scale = self.eye_distance_ / eye_distance
        tr = (self.eyes_mid_point_/scale - eyes_mid_point)
        tr = (int(tr[0]*scale), int(tr[1]*scale))

        tform = AffineTransform(scale=(scale, scale), rotation=0, shear=0,
                                translation=tr)
        h, w = self.target_height_, self.target_width_
        img_tr = warp(img, tform.inverse, output_shape=(h, w))
        return np.array(img_tr*255, dtype='uint8')




    # the only function in the class that gets an image path
    def align_to_ffhq(self, img_path, output_size=1024,
                    transform_size=4096, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(img_path), start=1):

            lm = np.array(face_landmarks)
            lm_chin = lm[0: 17]  # left-right
            lm_eyebrow_left = lm[17: 22]  # left-right
            lm_eyebrow_right = lm[22: 27]  # left-right
            lm_nose = lm[27: 31]  # top-down
            lm_nostrils = lm[31: 36]  # top-down
            lm_eye_left = lm[36: 42]  # left-clockwise
            lm_eye_right = lm[42: 48]  # left-clockwise
            lm_mouth_outer = lm[48: 60]  # left-clockwise
            lm_mouth_inner = lm[60: 68]  # left-clockwise

            # Calculate auxiliary vectors.
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            eye_avg = (eye_left + eye_right) * 0.5
            eye_to_eye = eye_right - eye_left
            mouth_left = lm_mouth_outer[0]
            mouth_right = lm_mouth_outer[6]
            mouth_avg = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg

            # Choose oriented crop rectangle.
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c = eye_avg + eye_to_mouth * 0.1
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            qsize = np.hypot(*x) * 2


            img = PIL.Image.open(img_path)

            # Shrink.
            shrink = int(np.floor(qsize / output_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(img.size[0]) / shrink)),
                        int(np.rint(float(img.size[1]) / shrink)))
                img = img.resize(rsize, PIL.Image.ANTIALIAS)
                quad /= shrink
                qsize /= shrink

            # Crop.
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
                    int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
                    min(crop[2] + border, img.size[0]),
                    min(crop[3] + border, img.size[1]))
            if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
                img = img.crop(crop)
                quad -= crop[0:2]

            # Pad.
            pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
                int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
            pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
                max(pad[2] - img.size[0] + border, 0),
                max(pad[3] - img.size[1] + border, 0))
            if enable_padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                img = np.pad(np.float32(img),
                            ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                h, w, _ = img.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                                np.float32(w - 1 - x) / pad[2]),
                                1.0 - np.minimum(np.float32(y) / pad[1],
                                                np.float32(h - 1 - y) / pad[3]))
                blur = qsize * 0.02
                img += (scipy.ndimage.gaussian_filter(img, [blur, blur,
                                                            0]) - img) * np.clip(
                    mask * 3.0 + 1.0, 0.0, 1.0)
                img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
                img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                        'RGB')
                quad += pad[:2]

            # Transform.
            img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                                (quad + 0.5).flatten(), PIL.Image.BILINEAR)
            if output_size < transform_size:
                img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
            
            # convert img to cv2
            img = np.array(img)
            return img[:, :, ::-1].copy()
