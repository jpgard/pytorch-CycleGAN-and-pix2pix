"""
A dataset of aligned volumes (3-D images).
"""

from data.image_folder import is_image_file
from data.base_dataset import BaseDataset, get_params #, get_transform
import util.spatial_transforms as spatial_transforms
import os
import re
from collections import defaultdict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def make_volume_dataset(dir, max_dataset_size=float("inf")):
    """
    Assemble a nested list of the slices in each observation.
    Adapted from data.image_folder.make_dataset().
    """
    images = []
    regex = re.compile("(\d+)_(\d+)\..*")
    dataset_images = defaultdict(list)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    # sorting first by dataset_id, then by z-slice
    images = sorted(images, key=lambda x: (int(re.match(regex, os.path.basename(x)).group(1)),
                                           int(re.match(regex, os.path.basename(x)).group(2))))
    for impath in images:
        bp = os.path.basename(impath)
        res = re.match(regex, bp)
        dataset_id = res.group(1)
        z = res.group(2)
        dataset_images[dataset_id].append(impath)
    if len(dataset_images) > max_dataset_size:
        raise NotImplementedError
    # return just a list of the z-slice filepaths for each dataset
    return list(dataset_images.items())


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    """
    Get the 3d versions of transforms. Modified from base_dataset.get_transform().
    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        # todo: need a 3D resize function
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            # perform random crop with scale of 1
            transform_list.append(spatial_transforms.MultiScaleRandomCrop(scales=(1,), size=opt.crop_size))
            # transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            raise NotImplementedError
            # transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        raise NotImplementedError
    #     transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        raise NotImplementedError
        # if params is None:
        #     transform_list.append(transforms.RandomHorizontalFlip())
        # elif params['flip']:
        #     transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        # todo: this may not work; need to convert the list of PIL images or arrays to a 3D tensor of shape (channels, h, w)
        transform_list += [spatial_transforms.ToTensor()]
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [spatial_transforms.Normalize((0.5,), (0.5,))]
            # transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [spatial_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            # transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class AlignedVolumeDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = make_volume_dataset(self.dir_AB, opt.max_dataset_size)  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __len__(self):
        return len(self.AB_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_slice_paths = self.AB_paths[index]
        A_slices = list()
        B_slices = list()
        for s in AB_slice_paths:
            AB = Image.open(AB_slice_paths).convert('RGB')
            # split AB image into A and B
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))
            A_slices.append(A)
            B_slices.append(B)
        # todo: check that this is correct concatenation axis for pytorch; should be channels-first (?)
        A = np.concatenate(A_slices, axis=0)
        B = np.concatenate(B_slices, axis=0)
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        # todo: check that transform is applied correctly to 3d images
        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_slice_paths, 'B_paths': AB_slice_paths}





