import torch
import lpips
from .base_model import BaseModel
from . import networks
import os


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_lpips', type=float, default=5.0, help='weight for LPIPS loss between fake_B and real_B')
            parser.add_argument('--lambda_lpips_A', type=float, default=0.0, help='weight for LPIPS loss between real_A and fake_B')
            parser.add_argument('--lambda_l2', type=float, default=1.0, help='weight for L2 loss')
            parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for GAN loss')
        
        # Add attention-related parameters
        parser.add_argument('--use_attention', action='store_true', help='use attention in the UnetGenerator innermost layer')
        parser.add_argument('--attention_heads', type=int, default=8, help='number of attention heads for MultiheadAttentionBlock')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_LPIPS', 'G_LPIPS_A', 'G_L2', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      use_attention=opt.use_attention, 
                                      attention_heads=opt.attention_heads)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # self.criterionLPIPS = lpips.LPIPS(net='vgg').to(self.device)
            self.criterionLPIPS = lpips.LPIPS(net='squeeze').to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_gan
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # L2 loss
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_l2
        # LPIPS loss between fake_B and real_B
        self.loss_G_LPIPS = self.criterionLPIPS(self.fake_B, self.real_B).mean() * self.opt.lambda_lpips
        # LPIPS loss between real_A and fake_B
        self.loss_G_LPIPS_A = self.criterionLPIPS(self.real_A, self.fake_B).mean() * self.opt.lambda_lpips_A
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2 + self.loss_G_LPIPS + self.loss_G_LPIPS_A
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    # def load_networks(self, epoch):
    #     """Load all the networks from the disk, with backward compatibility for models without attention.
        
    #     Parameters:
    #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    #     """
    #     for name in self.model_names:
    #         if isinstance(name, str):
    #             load_filename = '%s_net_%s.pth' % (epoch, name)
    #             load_path = os.path.join(self.save_dir, load_filename)
    #             net = getattr(self, 'net' + name)
    #             if isinstance(net, torch.nn.DataParallel):
    #                 net = net.module
    #             print('loading the model from %s' % load_path)
    #             # if you are using PyTorch newer than 0.4 (e.g., built from
    #             # GitHub source), you can remove str() on self.device
    #             try:
    #                 state_dict = torch.load(load_path, map_location=str(self.device))
    #                 if hasattr(state_dict, '_metadata'):
    #                     del state_dict._metadata
                    
    #                 # Special handling for backward compatibility with models without attention
    #                 if name == 'G' and hasattr(net, 'use_attention') and net.use_attention:
    #                     # If current model uses attention but loaded model doesn't have attention layers
    #                     missing_keys = []
    #                     for k in net.state_dict().keys():
    #                         if 'attention' in k and k not in state_dict:
    #                             missing_keys.append(k)
                        
    #                     if missing_keys:
    #                         print("Detected missing attention keys in checkpoint. Initializing attention layers.")
    #                         # Initialize the model with the state dict, ignoring missing keys
    #                         for key in list(state_dict.keys()):
    #                             self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                            
    #                         # Load what we can
    #                         net.load_state_dict(state_dict, strict=False)
    #                         return
                    
    #                 # patch InstanceNorm checkpoints prior to 0.4
    #                 for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    #                     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    
    #                 net.load_state_dict(state_dict)
    #             except Exception as e:
    #                 import traceback
    #                 traceback.print_exc()
    #                 continue

    # def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
    #     """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    #     key = keys[i]
    #     if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
    #         if module.__class__.__name__.startswith('InstanceNorm') and \
    #                 (key == 'running_mean' or key == 'running_var'):
    #             if getattr(module, key) is None:
    #                 state_dict.pop('.'.join(keys))
    #         if module.__class__.__name__.startswith('InstanceNorm') and \
    #            (key == 'num_batches_tracked'):
    #             state_dict.pop('.'.join(keys))
    #     else:
    #         self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)