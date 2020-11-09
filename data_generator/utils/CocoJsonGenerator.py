#!/usr/bin/env python3

import datetime
import json
import os
import sys
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools





INFO = {
    "description": "CelebA-HQ Dataset",
    "url": "https://github.com/switchablenorms/CelebAMask-HQ",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "CelebA",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "CelebA-HQ",
        "url": "https://github.com/switchablenorms/CelebAMask-HQ"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'skin',
    },
    {
        'id': 2,
        'name': 'face_mask',
    },
    {
        'id': 3,
        'name': 'hat',

    },
    {
        'id': 4,
        'name': 'l_brow',

    },
    {
        'id': 5,
        'name': 'r_brow',

    },
    {
        'id': 6,
        'name': 'l_eye',

    },
    {
        'id': 7,
        'name': 'r_eye',

    },
    {
        'id': 8,
        'name': 'nose',

    },
    {
        'id': 9,
        'name': 'u_lip',

    },
    {
        'id': 10,
        'name': 'l_lip',

    },
    {
        'id': 11,
        'name': 'mouth',

    },
    {
        'id': 12,
        'name': 'l_ear',

    },
    {
        'id': 13,
        'name': 'r_ear',

    },
    {
        'id': 14,
        'name': 'neck',

    },
    {
        'id': 15,
        'name': 'hair',

    },
    {
        'id': 16,
        'name': 'cloth',

    },
    {
        'id': 17,
        'name': 'ear_r',

    },
    {
        'id': 18,
        'name': 'eye_g',

    },

]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[
        0].zfill(5)
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(
        os.path.basename(f))[0])]

    return files


def main():
    ROOT_DIR = sys.argv[1]
    IMAGE_DIR = sys.argv[2]
    ANNOTATION_DIR = sys.argv[3]
    OUTPUT_NAME = sys.argv[4]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    x = os.walk(IMAGE_DIR)
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files,
                                                          image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    class_id = [x['id'] for x in CATEGORIES if
                                x['name'] in annotation_filename][0]

                    category_info = {'id': class_id,
                                     'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/{}.json'.format(ROOT_DIR, OUTPUT_NAME),
              'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()