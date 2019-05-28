
import random
import math
import numbers
import collections
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from data.base_dataset import __crop as crop
from data.base_dataset import __flip
try:
    import accimage
except ImportError:
    accimage = None

class ToTensor(transforms.ToTensor):
    """Convert an array of ``PIL Image``s or ``numpy.ndarray``s to tensor.

    Converts an array of D PIL Images or numpy.ndarrays each of shape (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x D x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, img_ary):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        # create a list of D tensors of shape C x H x W; these are the slices
        slice_tensor_stack = tuple(F.to_tensor(img) for img in img_ary)
        # stack the tensors along axis 1 so we have shape (C x D x H x W) as required by nn.Conv3d
        tensor = torch.stack(slice_tensor_stack, dim=1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(transforms.Normalize):
    """
    Normalize a volumetric image of shape (C x D x H x W) with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __call__(self, volume):
        # datapted from F.normalize() with inplace=False
        volume = volume.clone()
        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        volume.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return volume

class Flip:
    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img_ary):
        """
        This is an object-oriented implementation of the default behavior in aligned_dataset.get_transform() when flip is provided, which was:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
        """
        return [__flip(img, self.flip) for img in img_ary]


class RandomCrop(transforms.RandomCrop):
    """
    Apply the same random cropping to each element of an array.
    """

    def __call__(self, img_ary):
        t = transforms.RandomCrop(size=self.size)
        return [t(img) for img in img_ary]


class RandomCropWithPos(transforms.RandomCrop):
    def __init__(self, size, crop_pos, **kwargs):
        super(RandomCropWithPos, self).__init__(size, **kwargs)
        self.crop_pos = crop_pos

    def __call__(self, img_ary):
        """
        This is an object-oriented implementation of the default behavior in aligned_dataset.get_transform() when crop_pos is provided, which was:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
        """
        assert self.size[0] == self.size[1]
        t = transforms.Lambda(lambda img: crop(img, self.crop_pos, self.size[0]))
        return [t(img) for img in img_ary]


class Resize(transforms.Resize):
    """Resize the array of PIL Images to the given size.

        Args:
            size (sequence or int): Desired output size. If size is a sequence like
                (h, w), output size will be matched to this. If size is an int,
                smaller edge of the image will be matched to this number.
                i.e, if height > width, then image will be rescaled to
                (size * height / width, size)
            interpolation (int, optional): Desired interpolation. Default is
                ``PIL.Image.BILINEAR``
        """
    def __call__(self, img_ary):
        """
        Args:
            img_ary: iterable of PIL.Image objects to be scaled.

        Returns:
            iterable of same length as ary with rescaled images in corresponding positions.
        """
        t = transforms.Resize(self.size, self.interpolation)
        return [t(img) for img in img_ary]
