import numpy as np
import torch.nn as nn
import torch


class ImplicitNet(nn.Module):
    def __init__(self, d_in, dims, skip_in=[], geometric_init=True, radius_init=1, beta=100, d_out=1):
        super(ImplicitNet, self).__init__()
        dims = [d_in] + dims + [d_out]
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for layer in range(0, self.num_layers-1):
            if layer+1 in skip_in:
                out_dim = dims[layer+1] - d_in
                if out_dim < 0:
                    out_dim = dims[layer+1]
                    dims[layer+1] = out_dim + d_in
            else:
                out_dim = dims[layer+1]

            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2)/np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin_" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, input):
        x = input
        for layer in range(0, self.num_layers-1):
            lin = getattr(self, "lin_" + str(layer))
            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers-2:
                x = self.activation(x)

        return x


class SDF_Model(object):
    def __init__(self, NetCkp, device):
        layer = 8
        mdim = 512
        netdims = [mdim for _ in range(layer)]
        self.dim_geo = 3

        self.network = ImplicitNet(d_in=self.dim_geo, dims=netdims,
                                   skip_in=[4], geometric_init=True, radius_init=1, beta=100).to(device)

        self.device = device
        self.center = torch.randn(3)

        ckp = self.load_ckp(NetCkp)
        self.network.load_state_dict(ckp['network'])
        self.center.data = ckp['data_center'].data
        self.center = self.center.to(self.device)
        self.center = self.center.unsqueeze(0).unsqueeze(0)
        self.center.requires_grad = False
        #print(self.center.size())

        for param in self.network.parameters():
            param.requires_grad = False

    def sdf(self, x):
        x = x-self.center
        out = self.network(x)
        return out

    def load_ckp(self, fileName):
        ckp = torch.load(fileName, map_location=lambda storage, loc: storage)
        return ckp