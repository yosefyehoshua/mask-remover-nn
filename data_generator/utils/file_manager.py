import os
import shutil
import json
import tarfile
import zipfile
import bz2
import imageio

def check_exists(path):
    path = os.path.expanduser(path)
    return os.path.exists(path)

def makedir(path):
    path = os.path.expanduser(path)
    if not check_exists(path):
        os.makedirs(path)

def extract_file(path, to_directory=None):
    path = os.path.expanduser(path)
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith(('.tar.gz', '.tgz')):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith(('tar.bz2', '.tbz')):
        opener, mode = tarfile.open, 'r:bz2'
    elif path.endswith('.bz2'):
        opener, mode = bz2.BZ2File, 'rb'
        with open(path[:-4], 'wb') as fp_out, opener(path, 'rb') as fp_in:
            for data in iter(lambda: fp_in.read(100 * 1024), b''):
                fp_out.write(data)
        return
    else:
        raise (ValueError,
               "Could not extract `{}` as no extractor is found!".format(path))

    if to_directory is None:
        to_directory = os.path.abspath(os.path.join(path, os.path.pardir))
    cwd = os.getcwd()
    os.chdir(to_directory)

    try:
        file = opener(path, mode)
        try:
            file.extractall()
        finally:
            file.close()
    finally:
        os.chdir(cwd)


def move_file(file_path, move_to_path):
    shutil.move(file_path, move_to_path)


def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
    except OSError:
        print("Rename of the file %s failed" % old_name)
    else:
        print("Successfully Renamed the the file to: %s " % new_name)

def write_json_arr(obj_arr, out_pth):
    with open(out_pth, 'w') as outfile:
        json.dump([obj.__dict__ for obj in obj_arr], outfile)
        print(f'Saved json to {out_pth}')

def save(image, image_name, image_path):
    new_image_path = image_path + image_name
    image.save(new_image_path)
    print(f'Save to {new_image_path}')


def split_CelebA_dataset(root, img_path_train, mask_path_train,
                         img_path_val, mask_path_val, split=0.8):

    split = 15 * split  # split is limited to: 0.2, 0.4, 0.6, 0.8, 1.0

    atts = ['skin']  # more attributes can be added

    makedir(img_path_train)
    makedir(mask_path_train)

    makedir(img_path_val)
    makedir(mask_path_val)

    for i in range(s, e):

        if i < split:
            mask_path = mask_path_train
            img_path = img_path_train
        else:
            mask_path = mask_path_val
            img_path = img_path_val

        for j in range(i * 2000, (i + 1) * 2000):

            img_path_src = os.path.join(root, 'CelebAMask-HQ/CelebA-HQ-img',
                                     str(j), '.jpg')
            anno_path_dir = os.path.join(root, 'CelebAMask-HQ/CelebAMask-HQ-mask-anno')

            for l, att in enumerate(atts, 1):
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path_anno = os.path.join(anno_path_dir, str(i), file_name)

                if os.path.exists(path_anno):
                    shutil.copyfile(path_anno,
                                    '{}/{}.png'.format(mask_path, file_name))
                    print('saved {}.png'.format(file_name))
            try:
                shutil.copyfile(img_path_src, '{}/{}.jpg'.format(img_path, j))
            except OSError:
                print('copyfile of the file {}.jpg failed'.format(j))
            else:
                print('saved {}.jpg'.format(j))




