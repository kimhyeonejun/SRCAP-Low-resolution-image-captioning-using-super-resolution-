import torch
import numpy as np
import torch.nn as nn
import warnings
import utils as cfg
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")
warnings.filterwarnings("ignore", category=UserWarning, message="The attention mask and the pad token id were not set")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")

class GANLoss(nn.Module):
    """Define different GAN Discriminator's objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, loss_mode, which_net, which_D, target_real_label=1.0, target_fake_label=0.0, CUDA=True):
        """ Initialize the GAN's Discriminator Loss class.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_mode = loss_mode
        self.which_net = which_net
        self.which_D = which_D
        self.gpu = CUDA

        if loss_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_mode in ['vanilla', 'ragan', 'rsgan']:
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_mode in ['wgan', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % loss_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        if self.gpu:
            target_tensor = target_tensor.cuda()
        return target_tensor.expand_as(prediction)

    def G_loss(self, Dreal, Dfake):
        if self.loss_mode != 'rsgan' and cfg.d_out_mean:
            Dfake = torch.mean(Dfake.view(len(Dfake), -1), dim=-1)
            Dreal = torch.mean(Dreal.view(len(Dreal), -1), dim=-1)

        real_tensor = self.get_target_tensor(Dreal, True)
        fake_tensor = self.get_target_tensor(Dreal, False)

        if self.which_D == 'S':
            prediction_fake = Dfake
            prediction_real = real_tensor if self.loss_mode in ['vanilla'] else fake_tensor
        elif self.which_D == 'Ra':
            prediction_fake = Dfake - torch.mean(Dreal)
            prediction_real = Dreal - torch.mean(Dfake)
        else:
            raise NotImplementedError('which_D name [%s] is not recognized' % self.which_D)

        if self.loss_mode in ['lsgan', 'ragan']:
            loss_fake = self.loss(prediction_fake, real_tensor)
            loss_real = self.loss(prediction_real, fake_tensor)
            g_loss = loss_fake + loss_real
        elif self.loss_mode == 'vanilla':
            loss_fake = -self.loss(prediction_fake, fake_tensor)
            g_loss = loss_fake
        elif self.loss_mode in ['wgan', 'hinge'] and self.which_D == 'S':
            loss_fake = -prediction_fake.mean()
            loss_real = prediction_real.mean()
            g_loss = loss_fake + loss_real
        elif self.loss_mode == 'hinge' and self.which_D == 'Ra':
            loss_fake = nn.ReLU()(1.0 - prediction_fake).mean()
            loss_real = nn.ReLU()(1.0 + prediction_real).mean()
            g_loss = loss_fake + loss_real
        elif self.loss_mode == 'rsgan':
            loss_fake = self.loss(Dfake - Dreal, real_tensor)
            g_loss = loss_fake
        else:
            raise NotImplementedError('loss_mode name [%s] is not recognized' % self.loss_mode)

        return g_loss

    def D_loss(self, Dreal, Dfake):
        if self.loss_mode != 'rsgan' and cfg.d_out_mean:
            Dfake = torch.mean(Dfake.view(len(Dfake), -1), dim=-1)
            Dreal = torch.mean(Dreal.view(len(Dreal), -1), dim=-1)

        real_tensor = self.get_target_tensor(Dreal, True)
        fake_tensor = self.get_target_tensor(Dreal, False)

        if self.which_D == 'S':
            prediction_fake = Dfake
            prediction_real = Dreal
        elif self.which_D == 'Ra':
            prediction_fake = Dfake - torch.mean(Dreal)
            prediction_real = Dreal - torch.mean(Dfake)
        else:
            raise NotImplementedError('which_D name [%s] is not recognized' % self.which_D)

        if self.loss_mode in ['lsgan', 'ragan', 'vanilla']:
            loss_fake = self.loss(prediction_fake, fake_tensor)
            loss_real = self.loss(prediction_real, real_tensor)
        elif self.loss_mode == 'wgan':
            loss_fake = prediction_fake.mean()
            loss_real = -prediction_real.mean()
        elif self.loss_mode == 'hinge':
            loss_fake = nn.ReLU()(1.0 + prediction_fake).mean()
            loss_real = nn.ReLU()(1.0 - prediction_real).mean()
        elif self.loss_mode == 'rsgan':
            loss_fake = 0.
            loss_real = self.loss(Dreal - Dfake, real_tensor)
        else:
            raise NotImplementedError('loss_mode name [%s] is not recognized' % self.loss_mode)

        return loss_fake + loss_real

    def __call__(self, Dreal, Dfake):
        """Calculate loss given Discriminator's output and grount truth labels."""
        if self.which_net == 'G':
            return self.G_loss(Dreal, Dfake)
        elif self.which_net == 'D':
            return self.D_loss(Dreal, Dfake)
        else:
            raise NotImplementedError('which_net name [%s] is not recognized' % self.which_net)

class adversial_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, discrimination):
        #print(discrimination) #-> approaching zero
        targets = torch.ones_like(discrimination)
        loss = self.criterion(targets, discrimination)
        return loss
    
def G_loss(output, target_ids, img, img_real, discrimination):
  
    # Calculate the loss
    loss_fn = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    output = output.contiguous().float()
    target_ids = target_ids.contiguous().float()
    
    output= F.softmax(output, dim=1)
    target_ids = F.softmax(target_ids, dim=1)
    
    # Compute cross-entropy loss
    adversarial_loss_generated = adversial_loss()
    loss_fn = F.cross_entropy(output, target_ids)
    loss = loss_fn + mse_loss(img, img_real) * 0.1 + adversarial_loss_generated(discrimination)*0.001
    return loss