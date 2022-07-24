from models import *
from numpy import pi


def uvPoseEncoding(uvTensor, K):
    pEn = []
    for k in range(K):
        w = pi / (2. ** (k + 1))
        sinp = torch.sin(w * uvTensor)
        cosp = torch.cos(w * uvTensor)
        pEn.append(torch.cat([sinp, cosp], dim=1))
    pEn = torch.cat(pEn, dim=1)
    return pEn


class NeuralFeatMap(nn.Module):
    def __init__(self, device):
        super(NeuralFeatMap, self).__init__()
        self.device = device

    def forward(self, H, W, pX, pY, vertMask, hatF, jInfo=None):
        if jInfo is not None:
            vInfo = torch.cat([hatF, jInfo], dim=0)  # jInfo: dimFeat, numVerts
            out = torch.zeros(vInfo.size()[0], H, W).to(self.device)
            out[:, pY, pX] = torch.mm(vInfo, vertMask)
            return out
        else:
            out = torch.zeros(hatF.size()[0], H, W).to(self.device)
            out[:, pY, pX] = torch.mm(hatF, vertMask)  # [dimFeat, numVerts] * [numVerts, len(pX)] = [dimFeat, len(pX)]
            return out


def KL_loss(mu, log_sigma_2):
    loss = 0.5 * (mu**2 + torch.exp(log_sigma_2) - log_sigma_2 - 1)
    loss = torch.sum(loss, dim=1)
    return torch.mean(loss, dim=0)


def Gaussian_Regulatiozation(mu, sigma):
    '''
    :param mu: [b, dim]
    :param sigma: [b, dim]
    :return: scale
    '''
    loss = mu**2 + (sigma**2-1.)**2
    loss = torch.mean(torch.sum(loss, dim=-1), 0)
    return loss


def regulationZ(z, sigma):
    loss = z**2 / (sigma**2)
    loss = torch.abs(loss.mean() - 1.)
    return loss


def regularizeKernalR(r):
    loss = torch.var(r)
    return loss


def SampleMap(MapImg, MIndex, MEffi):
    '''
    MapImg [H, W, dim]
    MIndex [numV, 4, 2]
    MEffi [numV, 4]
    '''
    p0 = MapImg[MIndex[:, 0, 1], MIndex[:, 0, 0], :].unsqueeze(1)
    p1 = MapImg[MIndex[:, 1, 1], MIndex[:, 1, 0], :].unsqueeze(1)
    p2 = MapImg[MIndex[:, 2, 1], MIndex[:, 2, 0], :].unsqueeze(1)
    p3 = MapImg[MIndex[:, 3, 1], MIndex[:, 3, 0], :].unsqueeze(1)

    pp = torch.cat([p0, p1, p2, p3], dim=1)

    p = MEffi.unsqueeze(-1) * pp
    p = torch.sum(p, 1)
    return p


class RelativePosEncoder(nn.Module):
    def __init__(self, in_ch, out_ch=128):
        super(RelativePosEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(CNN2dLayer(in_ch, 512, 3, 1, 1),
                                       CNN2dLayer(512, 256, 3, 1, 1),
                                       CNN2dLayer(256, 128, 3, 1, 1),
                                       CNN2dLayer(128, out_ch, 3, 1, 1))  # 128 --> 128

    def forward(self, x):
        x = self.encoder_1(x)
        return x


class DisEncoder(nn.Module):
    def __init__(self, in_ch, out_ch=1024, midSize=64):
        super(DisEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(CNN2dLayer(in_ch, 128, 3, 2, 1),  # 64
                                       CNN2dLayer(128, 256, 3, 2, 1),  # 32
                                       CNN2dLayer(256, 512, 3, 2, 1),  # 16
                                       CNN2dLayer(512, 1024, 3, 2, 1),  # 8
                                       CNN2dLayer(1024, 1024, 3, 2, 1),  # 4
                                       CNN2dLayer(1024, out_ch, 3, 2, 1))  # 2

        self.in_ZLinear = nn.Linear(4*out_ch, midSize)

    def forward(self, x):
        x = self.encoder_1(x)

        b, c, h, w = x.size()
        z = torch.flatten(x, 1, -1)  # [b, 4*1024]
        z = self.in_ZLinear(z)

        return z, b, c, h, w


class DisDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DisDecoder, self).__init__()
        self.decoder1 = nn.Sequential(CNNTrans2dLayer(in_ch, 1024, 4, 2, 1),  # 4
                                      CNNTrans2dLayer(1024, 1024, 4, 2, 1),  # 8
                                      CNNTrans2dLayer(1024, 512, 4, 2, 1),  # 16
                                      CNNTrans2dLayer(512, 256, 4, 2, 1),  # 32
                                      CNNTrans2dLayer(256, 128, 4, 2, 1),  # 64
                                      CNNTrans2dLayer(128, 64, 4, 2, 1),  # 128
                                      CNNTrans2dLayer(64, out_ch, 4, 2, 1))  # 256

    def forward(self, x):
        x = self.decoder1(x)
        return x


class Pred_decoder(nn.Module):
    def __init__(self, in_ch, midSize, uv_ch, out_ch):
        super(Pred_decoder, self).__init__()
        #midSize = 512
        #midSize = 64
        #self.in_ZLinear = nn.Linear(in_ch * 4, midSize)
        self.out_ZLinear = LinearLayer(midSize, in_ch * 4)

        self.decoder = DisDecoder(in_ch, 64)

        self.uvencoder = nn.Sequential(LinearLayer(uv_ch, 64), LinearLayer(64, 64),
                                       LinearLayer(64, 64), LinearLayer(64, 64))

        self.MLP_pred = Dis_MLP(128, out_ch)
        self.out_ch = out_ch

    def forward(self, midz, uv, gMIndex, gMEffi, b, c, h, w):
        z = self.out_ZLinear(midz)
        z = z.view(b, c, h, w)

        z = self.decoder(z)
        z = z.permute(0, 2, 3, 1)  # [bsize, H, W, dim]

        fuv = self.uvencoder(uv)

        out_vp = []
        for bi in range(b):
            bz = SampleMap(z[bi, :, :, :], gMIndex, gMEffi)
            bz = torch.cat([bz, fuv], dim=-1)
            vp = self.MLP_pred(bz)
            out_vp.append(vp.unsqueeze(0))

        out_vp = torch.cat(out_vp, dim=0)
        return out_vp


class DynamicDeltaEncoder(nn.Module):
    def __init__(self, in_ch, v_ch, out_ch=128):
        super(DynamicDeltaEncoder, self).__init__()
        self.encoder_1 = nn.Sequential(CNN2dLayer(in_ch, 512, 3, 1, 1),
                                       CNN2dLayer(512, 256, 3, 1, 1),
                                       CNN2dLayer(256, 128, 3, 1, 1))  # 128 --> 128
        self.encoder_2 = nn.Sequential(CNN2dLayer(128+v_ch, 128, 3, 1, 1),
                                       CNN2dLayer(128, 128, 3, 1, 1),
                                       CNN2dLayer(128, out_ch, 3, 1, 1))

    def forward(self, f, v):
        x = self.encoder_1(f)
        x = self.encoder_2(torch.cat([x, v], dim=1))
        return x


def calcW(pa, radiuseSigma):
    '''
    :param pa: [b, N, k]
    :param radiuseSigma: [k]
    :return: [b, N, k]
    '''

    rr = radiuseSigma.unsqueeze(0).unsqueeze(0)  # [1, 1, k]
    ra = -(pa**2)/(2.*rr**2)
    W = torch.softmax(ra, dim=-1)

    return W


