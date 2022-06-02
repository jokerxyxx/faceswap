import torch
import torch.nn as nn
from . import demos


class BackModel(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, timestep_dim, n_blocks=6, 
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(BackModel, self).__init__()

        activation = nn.ReLU(True)
        self.attention1 = demos.SELayer(128)
        self.attention2 = demos.SELayer(256)
        self.attention3 = demos.SELayer(256)
        self.attention4 = demos.SELayer(128)
        
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)  
        #64*224*224
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.embed1 = demos.ResnetBlock_Adain(128,latent_size=timestep_dim,padding_type=padding_type,activation=activation)
        #128*112*112
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.embed2 = demos.ResnetBlock_Adain(256,latent_size=timestep_dim,padding_type=padding_type,activation=activation)
        #256*56*56
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
        #512*28*28
        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                demos.ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.embed3 = demos.ResnetBlock_Adain(256,latent_size=timestep_dim,padding_type=padding_type,activation=activation)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.embed4 = demos.ResnetBlock_Adain(128,latent_size=timestep_dim,padding_type=padding_type,activation=activation)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))

    def forward(self, input, dlatents, timestep):
        x = input  # 3*224*224
        dim = demos.timestep_embedding(timestep,64)
        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip2 = self.embed1(skip2,dim)
        skip2 = self.attention1(skip2)
        skip3 = self.down2(skip2)
        skip3 = self.embed2(skip3,dim)
        skip3 = self.attention2(skip3)
        x = self.down3(skip3)


        bot = []
        bot.append(x)
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        x = self.up3(x)
        x = self.embed3(x,dim)
        x = self.attention3(x)
        x = self.up2(x)
        x = self.embed4(x,dim)
        x = self.attention4(x)
        x = self.up1(x)
        x = self.last_layer(x)
        # x = (x + 1) / 2

        # return x, bot, features, dlatents
        return x

