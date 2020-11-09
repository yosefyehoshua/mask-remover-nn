import os
import numpy as np
import face_recognition
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, celeba_img, mask_path, show=False, model='hog'):
        self.celeba_img = celeba_img
        self.face_path = celeba_img.img_path
        self.skin_anno_path = celeba_img.gt_dict['skin']
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._skin_anno: ImageFile = None
        self._mask_img: ImageFile = None
        self._mask_anno: ImageFile = None  # mask segmentation of face_img
        self._is_face: bool = False

    def mask(self):

        face_image_np = face_recognition.load_image_file(self.face_path)

        face_locations = face_recognition.face_locations(face_image_np,
                                                         model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np,
                                                         face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._face_img.info['image_name'] = os.path.basename(self.face_path)

        self._skin_anno = Image.open(self.skin_anno_path)
        self._skin_anno.info['image_name'] = os.path.basename(self.skin_anno_path)

        self._mask_img = Image.open(self.mask_path)
        # cropping image padding
        imageBox = self._mask_img.getbbox()
        self._mask_img = self._mask_img.crop(imageBox)
        self._mask_img.info['image_name'] = os.path.basename(self.mask_path)

        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            self._is_face = True
            self._mask_face(face_landmark)

        if self._is_face:
            if self.show:
                self._face_img.show()

            self._save()
        else:
            print('Found no face.')

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point,
                                                               nose_point,
                                                               chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(
            chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0),
                       mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1],
                           chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x0 = np.abs(center_x + int(
            offset * np.cos(radian)) - rotated_mask_img.width // 2)
        box_y0 = center_y + int(
            offset * np.sin(radian)) - rotated_mask_img.height // 2
        box_x0 = np.abs(box_x0)


        mask_img = mask_img.convert('RGB')

        # resizing skin_annotation for np array substraction
        img_resize = self._skin_anno.resize((1024, 1024), Image.ANTIALIAS)

        skin_anno_arr = np.array(img_resize)

        img_arr = np.array(mask_img)
        skin_arr_bool = np.array(skin_anno_arr, dtype=bool)

        skin_anno_arr.flags.writeable = True
        img_arr.flags.writeable = True


        box_x1 = box_x0 + mask_img.size[0]
        box_y1 = box_y0 + mask_img.size[1]

        if (box_x0 + mask_img.size[0] > self._face_img.size[0]):
            img_arr = img_arr[:,:skin_arr_bool.shape[1] - box_x0]
        if (box_y0 + mask_img.size[1] > self._face_img.size[1]):
            img_arr = img_arr[:skin_arr_bool.shape[0] - box_y0,:]

        # adjust face mask image to skin annotation
        img_arr *= skin_arr_bool[box_y0: box_y1, box_x0: box_x1]

        img_arr = self.make_rgba_transparent(img_arr)

        # add mask
        self._face_img.paste(img_arr, (box_x0, box_y0), img_arr)

        from facemasker_helper import create_binary_mask

        # create mask annotation
        self._mask_anno = create_binary_mask(
            (self._face_img.size[1], self._face_img.size[0]), img_arr,
            (box_x0, box_y0)).resize((512, 512), Image.ANTIALIAS)





    @staticmethod
    def make_rgba_transparent(img_arr):
        img_arr = Image.fromarray(img_arr)
        img_arr = img_arr.convert('RGBA')

        datas = img_arr.getdata()

        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)

        img_arr.putdata(newData)
        return img_arr

    @staticmethod
    def drawBB(img, BBox):
        x0, y0, w, h = BBox
        im = np.ascontiguousarray(img)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        # Create a Rectangle patch
        rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()

    def _save(self):


        anno_name = os.path.splitext(os.path.basename(self.face_path))[0].zfill(5) + '_face_mask'

        mask_path_splits = os.path.split(self.skin_anno_path)
        base_name_suffix_splits = os.path.splitext(mask_path_splits[1])
        new_mask_face_path = mask_path_splits[0] + '/' + anno_name + base_name_suffix_splits[1]

        try:
            self._mask_anno.save(new_mask_face_path)
            self._face_img.save(self.face_path)
            self.celeba_img.add_gt_to_dict(new_mask_face_path)
        except OSError:
            print("failed to save files")
        else:
            print(f'Saved: {os.path.basename(new_mask_face_path)} and {os.path.basename(self.face_path)}')


    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (
                               line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (
                                       line_point1[0] - line_point2[0]))
        return int(distance)

    def is_face(self):
        return self._is_face