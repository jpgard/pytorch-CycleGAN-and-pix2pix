"""

"""

from data.base_dataset import BaseDataset
from data.image_folder import is_image_file
import os
import re
from collections import defaultdict

def make_volume_dataset(dir, max_dataset_size=float("inf")) -> dict:
    """
    Assemble a nested list of the slices in each observation.
    Adapted from data.image_folder.make_dataset()
    """
    images = []
    dataset_images = defaultdict(list)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # todo: combine fnames with common dataset_id prefix; this should return a NESTED list where each element is a list of the files for a given dataset.
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    # sorting the filenames will yield a list sorted first by dataset_id, then by z-slice
    images = sorted(images)
    for impath in images:
        bp = os.path.basename(impath)
        res = re.match("(\d+)_(\d+)\..*", bp)
        dataset_id = res.group(1)
        z = res.group(2)
        dataset_images[dataset_id].append(impath)
        assert len(dataset_images[dataset_id]) == z + 1, "missing z-slice in input data"
    if len(dataset_images) > max_dataset_size:
        raise NotImplementedError
    return dataset_images


class AlignedVolumeDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = make_volume_dataset(self.dir_AB, opt.max_dataset_size)  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __len__(self):
        # todo
        pass

    def __getitem__(self, item):
    """Return a data point and its metadata information.

    Parameters:
        index - - a random integer for data indexing

    Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor) - - an image in the input domain
        B (tensor) - - its corresponding image in the target domain
        A_paths (str) - - image paths
        B_paths (str) - - image paths (same as A_paths)
    """
        # todo
        pass

    def modify_commandline_options(parser, is_train):
        # todo
        pass




