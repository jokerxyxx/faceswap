import torch
import torch.nn as nn

from .base_model import BaseModel
from .back_model import BackModel

class fsModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint['model'].module
        self.netArc = self.netArc.cuda()
        self.netArc.eval()
        self.netArc.requires_grad_(False)

        #diffusion network
        self.netBack = BackModel(input_nc=3,output_nc=3,latent_size=512,timestep_dim=64)
        self.netBack.cuda()
        
        params = list(self.netBack.parameters())
        self.optimizer_Back = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print (pretrained_path)
            self.load_network(self.netBack, opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_Back, opt.which_epoch, pretrained_path)

        torch.cuda.empty_cache()
            
    def save(self, which_epoch):
        self.save_network(self.netBack,  which_epoch)
        self.save_optim(self.optimizer_Back,  which_epoch)