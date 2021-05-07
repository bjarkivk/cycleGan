import torch
import models
from Base_Model import BaseModel
import itertools

class CycleGANModel(BaseModel):

    def __init__(self, opt):

        #BaseModel.__init__(self, opt)
        self.losses = ['D_1', 'G_1', 'D_2', 'G_2','cycle_1','cycle_2']
        # define models
        self.generator_X = models.Generator(3, 3, 64, use_dropout=False,  init_gain=0.02, gpu_ids=[])
        self.generator_Y = models.Generator(3, 3, 64, use_dropout=False, init_gain=0.02, gpu_ids=[])
        self.discriminator_X = models.Discriminator(3, 64, n_layers_D=3, init_gain=0.02, gpu_ids=[])
        self.discriminator_Y = models.Discriminator(3, 64, n_layers_D=3, init_gain=0.02, gpu_ids=[])
        
        # define loss functions
        self.ganLoss = models.GANLoss('lsgan')
        self.cycleLoss = torch.nn.L1Loss()
        # define optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.generator_X.parameters(), self.generator_Y.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.discriminator_X.parameters(), self.discriminator_Y.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


    def forward(self):
        # forward mapping
        self.fake_Y = self.generator_X(self.real_X)
        self.fake_X = self.generator_Y(self.real_Y)
        # backward mapping
        self.re_X = self.generator_Y(self.fake_Y)
        self.re_Y = self.generator_X(self.fake_X)
        
    def backward_D(self):
        # loss for real
        real_score_X = discriminator_X(self.real_X)
        loss_real_X = self.ganLoss(real_score_X)

        real_score_Y = discriminator_Y(self.real_Y)
        loss_real_Y = self.ganLoss(real_score_Y)

        # loss for fake
        fake_score_X = discriminator_X(self.fake_X)
        loss_fake_X = self.ganLoss(fake_score_X)

        fake_score_Y = discriminator_Y(self.fake_Y)
        loss_real_Y = self.ganLoss(fake_score_Y)

        # combined loss & calculate gradients
        loss_X = (loss_real_X + loss_fake_X)
        loss_X.backward()
        loss_Y = (loss_real_Y + loss_fake_Y)
        loss_Y.backward()

    def backward_G(self):

        self.loss_generator_X = self.ganLoss(self.discriminator_Y(self.fake_Y))
        self.loss_generator_Y = self.ganLoss(self.discriminator_X(self.fake_X))

        self.loss_cycle_X = self.cycleLoss(self.re_X, self.real_A)
        self.loss_cycle_Y = self.cycleLoss(self.re_Y, self.real_Y)
        
        # combined loss and calculate gradients
        self.loss = self.loss_generator_X + self.loss_generator_Y + self.loss_cycle_X + self.loss_cycle_Y
        self.loss.backward()

    def optimize_parameters(self):
        """called this at each training iteration"""
        # generate fake and reconstructed images.
        self.forward()

        # TO DO : fix D when optimizing G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
