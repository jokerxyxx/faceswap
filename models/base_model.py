import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass
    
    # helper saving function that can be used by subclasses
    def save_network(self, network, epoch_label, gpu_ids=None):
        save_filename = '{}_net.pth'.format(epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda()

    def save_optim(self, network, epoch_label, gpu_ids=None):
        save_filename = '{}_optim.pth'.format(epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)


    # helper loading function that can be used by subclasses
    def load_network(self, network, epoch_label, save_dir=''):        
        save_filename = '%s_net.pth' % (epoch_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)

        else:
            network.load_state_dict(torch.load(save_path))
            
    # helper loading function that can be used by subclasses
    def load_optim(self, network, epoch_label, save_dir=''):        
        save_filename = '%s_optim.pth' % (epoch_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
        else:
            network.load_state_dict(torch.load(save_path))
                            

    def update_learning_rate():
        pass
