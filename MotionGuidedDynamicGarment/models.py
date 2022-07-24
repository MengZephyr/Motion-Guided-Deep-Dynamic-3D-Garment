import torch.nn as nn
import torch
import math


class CNN2dLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding, k_relu=0.2):
        super(CNN2dLayer, self).__init__()
        if k_relu < 0:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                      nn.InstanceNorm2d(out_ch),
                                      nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                      nn.InstanceNorm2d(out_ch),
                                      nn.LeakyReLU(k_relu, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class CNNTrans2dLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding, k_relu=0.2):
        super(CNNTrans2dLayer, self).__init__()
        if k_relu < 0:
            self.transConv = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                           nn.InstanceNorm2d(out_ch),
                                           nn.ReLU())
        else:
            self.transConv = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                           nn.InstanceNorm2d(out_ch),
                                           nn.LeakyReLU(k_relu, inplace=True))

    def forward(self, x):
        x = self.transConv(x)
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k_relu=0.01, ifNorm=False):
        super(LinearLayer, self).__init__()
        if k_relu > 0:
            self.FC = nn.Sequential(nn.Linear(in_ch, out_ch), nn.LeakyReLU(k_relu, inplace=True))
        else:
            self.FC = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(inplace=True))

        if ifNorm:
            self.midlayerNorm = nn.LayerNorm(out_ch, elementwise_affine=False)
        self.ifNorm = ifNorm

    def forward(self, x, ifLinear = False):
        if ifLinear:
            x = self.FC[0](x)
        else:
            x = self.FC(x)
        if self.ifNorm:
            x = self.midlayerNorm(x)

        return x


class miniSimpleGaussian(torch.nn.Module):
    def __init__(self, mid_ch):
        super(miniSimpleGaussian, self).__init__()
        self.en_mu = nn.Linear(mid_ch, mid_ch)
        self.en_log_sigma = nn.Linear(mid_ch, mid_ch)

    def forward(self, x, std_z=None):
        mu = self.en_mu(x)
        log_sigma_2 = self.en_log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma_2)
        if std_z is None:
            std_z = torch.randn_like(sigma).to(sigma)
        z = mu + sigma * std_z
        return z, mu, sigma


class miniVAE(torch.nn.Module):
    def __init__(self, mid_ch):
        super(miniVAE, self).__init__()
        self.en_mu = nn.Linear(mid_ch, mid_ch)
        self.en_log_sigma = nn.Linear(mid_ch, mid_ch)

    def sample_latent(self, feat, std_z):
        mu = self.en_mu(feat)
        log_sigma_2 = self.en_log_sigma(feat)
        sigma = torch.exp(0.5 * log_sigma_2)
        if std_z is None:
            std_z = torch.randn_like(sigma).to(sigma)
        z = mu + sigma * std_z
        return z, mu, log_sigma_2

    def forward(self, x, std_z=None):
        z, mu, log_sigma_2 = self.sample_latent(x, std_z)
        return z, mu, log_sigma_2


class Res_MLP(nn.Module):
    def __init__(self, in_ch):
        super(Res_MLP, self).__init__()
        self.MLP = LinearLayer(in_ch, in_ch)

    def forward(self, x):
        x = x + self.MLP(x)
        return x


class Dis_MLP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dis_MLP, self).__init__()

        self.MLP_B = nn.Sequential(LinearLayer(in_ch, 64),
                                   Res_MLP(64), Res_MLP(64), Res_MLP(64),
                                   Res_MLP(64))
        self.LinearGeo = nn.Sequential(LinearLayer(64+in_ch, 64), nn.Linear(64, out_ch))

    def forward(self, x):
        z = self.MLP_B(x)
        vp = self.LinearGeo(torch.cat([z, x], dim=-1))
        return vp





