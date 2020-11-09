import re
import os


class CelebAImage:

    def __init__(self, img_path, gt_path):
        self.img_path = img_path

        gt_name = self._get_chars_from_str_path(gt_path)

        self.gt_dict = {gt_name: gt_path}

        self.gt_dir = os.path.dirname(gt_path)

        image_name_split = os.path.splitext(os.path.basename(self.img_path))

        self.img_index = int(image_name_split[0])

        self.suffix = image_name_split[1]

        self.merged_gt = None

    def get_image_idx(self):
        return self.img_index

    def has_mask(self):
        if 'face_mask' in self.gt_dict.keys():
            return True
        else:
            return False

    def get_img_path(self):
        return self.img_path

    def add_gt_to_dict(self, path):
        gt_name = self._get_chars_from_str_path(path)
        self.gt_dict[gt_name] = path

    def _get_chars_from_str_path(self, path):
        image_name_split = os.path.splitext(os.path.basename(path))
        name = "_".join(re.findall("[a-zA-Z]+", image_name_split[0]))
        return name

    def set_merge_gt(self, path):
        self.merged_gt = path


def create_celeba_arr(img_dir, anno_dir):
    celeba_arr: CelebAImage = []

    for i in range(15):

        atts = ['skin', 'face_mask']  # delete 'face_mask' after debug

        for j in range(i * 2000, (i + 1) * 2000):

            img_path_src = os.path.join(img_dir,
                                        str(j) + '.jpg')

            for att in atts:
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path_anno = os.path.join(anno_dir, str(i), file_name)

                if os.path.exists(path_anno):
                    if atts[0] == att:  # delete after debug, 'skin' must
                        # be first!
                        celeba_arr.append(CelebAImage(img_path_src, path_anno))
                    else:
                        celeba_arr[j].add_gt_to_dict(path_anno)

    return celeba_arr
