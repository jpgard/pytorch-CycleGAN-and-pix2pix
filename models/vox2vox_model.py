"""
The vox2vox model for working with 3D (volumetric) inputs.
"""

from models.pix2pix_model import Pix2PixModel
from models.base_model import BaseModel
import models.networks_3d as networks_3d
import models.networks as networks
import torch

class Vox2VoxModel(BaseModel):
    """ This class implements the vox2vox model, for learning a mapping from input volumes to output volumes given paired data.
    This is designed to mostly follow the default architecture of the pix2pix model, with appropriate modifications
    for the 3D case.

    The model training requires '--dataset_mode aligned_volume' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    see also models.pix2pix_model.py
    """
    def __init__(self, opt):
        """Initialize the vox2vox class. Modified from models.pix2pix_model .

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1L2', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks_3d.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks_3d.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if opt.loss_norm == "L1":
                self.criterionL1L2 = torch.nn.L1Loss()
            elif opt.loss_norm == "L2":
                self.criterionL1L2 = torch.nn.MSELoss()
            else:
                raise NotImplementedError("specify either an L1 or L2 as the loss_norm")
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # todo
        raise NotImplementedError

    def optimize_parameters(self):
        # todo
        raise NotImplementedError

    def forward(self):
        # todo
        raise NotImplementedError

