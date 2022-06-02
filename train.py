import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from models.faceswap_model import fsModel
from dataset import GetLoader
from utils import utils,plot,resizer
from models.sampler import Sampler,create_gaussian_diffusion
from models.demos import timestep_embedding
def str2bool(v):
    return v.lower() in ('true')

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='faceswap', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size') 
        self.parser.add_argument('--down_N', type=int, default=4, help='input downsample scale')       
        self.parser.add_argument('--range_t', type=int, default=0, help='from which to insert')   
        self.parser.add_argument('--dataset', type=str, default="./vggface2_crop_arcfacealign_224", help='path to the face swapping dataset')

        # for training
        self.parser.add_argument('--continue_train', type=str2bool, default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/faceswap', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='10000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')

        self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--total_step", type=int, default=1000000, help='total training step')
        self.parser.add_argument("--log_frep", type=int, default=10, help='frequence for printing log information')
        self.parser.add_argument("--sample_freq", type=int, default=20, help='frequence for sampling')
        self.parser.add_argument("--model_freq", type=int, default=10000, help='frequence for saving the model')

        self.isTrain = True
        
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            utils.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__ == '__main__':
    opt = TrainOptions().parse(save=True)
    cudnn.benchmark = True
    train_loader   =  GetLoader(opt.dataset,opt.batchSize,2,1234)
    
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    if not opt.continue_train:
        start   = 0
    else:
        start   = int(opt.which_epoch)
    total_step  = opt.total_step

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    model = fsModel()
    model.initialize(opt)
    diffusion = create_gaussian_diffusion()
    data_sampler = Sampler(diffusion=diffusion)
    optimizer_Back  = model.optimizer_Back

    shape = (opt.batchSize, 3, 224, 224)
    shape_d = (opt.batchSize, 3, int(224 / opt.down_N), int(224 / opt.down_N))
    down = resizer.Resizer(shape, 1 / opt.down_N).to(next(model.parameters()).device)
    up = resizer.Resizer(shape_d, opt.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    for step in range(start, total_step):
        model.netBack.train()
        src_image = train_loader.next()

        img_id_112      = F.interpolate(src_image[0:1],size=(112,112), mode='bicubic')
        latent_id       = model.netArc(img_id_112)
        latent_id       = F.normalize(latent_id, p=2, dim=1)
        
        t,weights = data_sampler.sample(opt.batchSize,device=torch.device(f"cuda"))
        dim = timestep_embedding(t,64)
        losses = diffusion.training_losses(model.netBack,src_image,t,latent_id,dim)
        loss = (losses["loss"] * weights).mean()

        optimizer_Back.zero_grad()
        loss.backward()
        optimizer_Back.step()


        if (step + 1) % opt.log_frep == 0:

            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % loss)
        ### display output images
        if (step + 1) % opt.sample_freq == 0:
            model.netBack.eval()
            with torch.no_grad():
                imgs        = list()
                zero_img    = (torch.zeros_like(src_image[0,...]))
                imgs.append(zero_img.cpu().numpy())
                save_img    = ((src_image.cpu())* imagenet_std + imagenet_mean).numpy()
                for r in range(opt.batchSize):
                    imgs.append(save_img[r,...])
                arcface_112     = F.interpolate(src_image,size=(112,112), mode='bicubic')
                id_vector_src  = model.netArc(arcface_112)
                id_vector_src  = F.normalize(id_vector_src, p=2, dim=1)

                for i in range(opt.batchSize):
                    
                    imgs.append(save_img[i,...])
                    image_infer = src_image[i, ...].repeat(opt.batchSize, 1, 1, 1)
                    img_fake    = diffusion.p_sample_loop(
                        model.netBack,
                        (opt.batchSize,3,224,224),
                        image_infer,
                        id_vector_src,
                        resizers=resizers,
                        range_t=opt.range_t
                    ).cpu()
                    
                    img_fake    = img_fake * imagenet_std
                    img_fake    = img_fake + imagenet_mean
                    img_fake    = img_fake.numpy()
                    for j in range(opt.batchSize):
                        imgs.append(img_fake[j,...])
                print("Save test data")
                imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
                plot.plot_batch(imgs, os.path.join(sample_path, 'step_'+str(step+1)+'.jpg'))

        ### save latest model
        if (step+1) % opt.model_freq==0:
            print('saving the latest model (steps %d)' % (step+1))
            model.save(step+1)   