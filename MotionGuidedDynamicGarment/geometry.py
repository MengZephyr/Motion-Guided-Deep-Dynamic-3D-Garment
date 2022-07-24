import torch
import numpy as np


def get_edges(faceIDs):
    edges = np.concatenate([faceIDs[:, 0:2], faceIDs[:, 1:3], np.stack([faceIDs[:, 2], faceIDs[:, 0]], axis=1)], axis=0)
    edges = np.sort(edges, axis=-1)
    edges = np.unique(edges, axis=0)
    return edges


def getCCEdges(ccMap):
    ncc = []
    for cc in ccMap:
        lenC = len(cc)
        if lenC > 1:
            for i in range(lenC):
                for j in range(i+1, lenC):
                    ncc.append([cc[i], cc[j]])
    ncc = np.array(ncc)
    ncc = np.sort(ncc, axis=-1)
    ncc = np.unique(ncc, axis=0)
    return ncc


def getLaplacianMatrix(edgeID, numVs):
    lapMatrix = torch.zeros((numVs, numVs), dtype=torch.float, requires_grad=False)
    lapMatrix[list(edgeID[:, 0]), list(edgeID[:, 1])] = 1.
    lapMatrix[list(edgeID[:, 1]), list(edgeID[:, 0])] = 1.
    sumL = torch.sum(lapMatrix, dim=1)
    sumL = torch.tensor(-1.)/sumL
    esumL = torch.diag(sumL)
    lapMatrix = torch.matmul(esumL, lapMatrix) + torch.eye(numVs)

    return lapMatrix


def normalize_vetors(v):
    s = torch.norm(v, dim=-1).unsqueeze(-1)
    v = v/s
    return v


def samplePnts(SamplevIDs, SampleUVs, verts, norms):
    samVs = []
    samNs = []
    for i in range(len(SamplevIDs)):
        vIDs = SamplevIDs[i]
        kuvs = SampleUVs[i]
        p = (1. - kuvs[0] - kuvs[1]) * verts[vIDs[0], :] + kuvs[0] * verts[vIDs[1], :] + kuvs[1] * verts[vIDs[2], :]
        samVs.append(p.unsqueeze(0))
        if norms is not None:
            n = (1. - kuvs[0] - kuvs[1]) * norms[vIDs[0], :] + kuvs[0] * norms[vIDs[1], :] + kuvs[1] * norms[vIDs[2], :]
            samNs.append(n.unsqueeze(0))

    samVs = torch.cat(samVs, dim=0)
    if norms is not None:
        samNs = torch.cat(samNs, dim=0)

    return samVs, samNs


def Relative_PointPosition(pnts, joints):
    '''
    pnts: [N, 3],
    joints: [k, 3],
    output [N, k*3], [N, k]
    '''
    k = joints.size()[0]
    px = pnts.unsqueeze(1)
    px = px.repeat(1, k, 1)   # [N, k, 3]
    jy = joints.unsqueeze(0)  # [1, k, 3]
    px = px - jy  # [N, k, 3]
    ax = torch.norm(px, dim=-1)  # [N, k]
    px = torch.flatten(px, 1, 2)  # [N, k*3]

    return ax, px


def compute_edgeLength(pnts, edgesID):
    v0 = pnts[:, list(edgesID[:, 0]), :]
    v1 = pnts[:, list(edgesID[:, 1]), :]
    ve = v0 - v1
    el = torch.norm(ve, dim=-1)
    return el


def projectD(GTV, GBV, GNV, D):
    dx = D[:, 0].unsqueeze(-1)  # [numV, 1]
    dy = D[:, 1].unsqueeze(-1)
    dz = D[:, 2].unsqueeze(-1)
    Dp = dx * GTV + dy * GBV + dz * GNV # [numV, 3]
    return Dp


def batchProjD(GTV, GBV, GNV, D, bsize, device):
    dd = torch.zeros_like(D).to(device)
    for b in range(bsize):
        d = D[b, :, :]
        tv = GTV[b, :, :]
        bv = GBV[b, :, :]
        nv = GNV[b, :, :]
        dd[b, :, :] = projectD(tv, bv, nv, d)
    return dd


def Local_GeoDeform(G0, GD, G0TV, G0BV, G0NV, bsize, device):
    resDD = batchProjD(GTV=G0TV, GBV=G0BV, GNV=G0NV, D=GD, bsize=bsize, device=device)
    DD = G0.unsqueeze(0) + resDD
    return DD


def computeBlendingRT(B0, R, T, W):
    '''
    :param B0: [numV, numB, 3]
    :param R: [numV, numB, 3, 3]
    :param T: [numV, numB, 3]
    :param W: [numV, numB, 1]
    :return:
    '''
    GR = W.unsqueeze(-1) * R
    GR = torch.sum(GR, 1)  # [numV, 3, 3]

    I = torch.eye(3).to(R.get_device())
    GT = I.unsqueeze(0).unsqueeze(0) - R  # [numV, numB, 3, 3]
    GT = torch.matmul(GT, B0.unsqueeze(-1))  # [numV, numB, 3, 1]
    GT = T + GT.squeeze(-1)  # [numV, numB, 3]
    GT = W * GT  # [numV, numB, 3]
    GT = torch.sum(GT, 1)  # [numV, 3]

    return GR, GT


def transformGX(x, R, T):
    '''
    :param x: [numV, 3]
    :param R: [numV, 3, 3]
    :param T: [numV, 3]
    :return:
    '''
    x = torch.matmul(R, x.unsqueeze(-1)).squeeze(-1) # [numV, 3]
    x = x + T
    return x


def batch_DeformGeo(CG, B0, BR, BT, BW, bsize, numV, device):
    out = torch.zeros(bsize, numV, 3).to(device)
    for b in range(bsize):
        G = CG[b, :, :]
        R = BR[b, :, :, :]
        R = R.unsqueeze(0).repeat(numV, 1, 1, 1)  # [numV, numB, 3, 3]
        T = BT[b, :, :]
        T = T.unsqueeze(0).repeat(numV, 1, 1)  # [numV, numB, 3]
        W = BW[b, :, :]
        W = W.unsqueeze(-1)

        GR, GT = computeBlendingRT(B0, R, T, W)
        iG = transformGX(G, GR, GT)

        out[b, :, :] = iG

    return out


def Global_GeoDeform(DD, B0, BR, BT, GW, bsize, numV, device):
    GG = batch_DeformGeo(CG=DD, B0=B0, BR=BR, BT=BT, BW=GW, bsize=bsize, numV=numV, device=device)
    return GG


def skewVector(v, numV, device):
    U = torch.zeros(numV, 3, 3).to(device)
    U[:, 0, 1] = -v[:, 2]
    U[:, 0, 2] = v[:, 1]
    U[:, 1, 0] = v[:, 2]
    U[:, 1, 2] = -v[:, 0]
    U[:, 2, 0] = -v[:, 1]
    U[:, 2, 1] = v[:, 0]
    return U


def Rotation_A_to_B(a, b, numV, device):
    '''
    a: [numV, 3], b: [numV, 3]
    '''
    c = torch.matmul(a.unsqueeze(1), b.unsqueeze(-1))  # a.dot(b) cos(\theta) [numV, 1, 1]
    v = torch.cross(a, b, dim=1)  # a.cross(b) [numV, 3]
    s = torch.norm(v, dim=1).unsqueeze(-1)  # sin(\theta) [numV, 1]
    id_zeros = torch.nonzero(s.view(-1) < 1.e-8)
    v = v / s
    v[id_zeros[:, 0], :] = 0.
    #print(torch.nonzero(torch.isnan(v.view(-1))))
    #exit(1)
    U = skewVector(v, numV, device)  # [numV, 3, 3]
    UU = torch.matmul(v.unsqueeze(-1), v.unsqueeze(1))  # [numV, 3, 3]
    s = s.unsqueeze(-1) # sin(\theta) [numV, 1, 1]
    E = torch.eye(3).unsqueeze(0)
    E = E.repeat(numV, 1, 1).to(device)  # [numV, 3, 3]

    R = c*E + s*U + (torch.ones_like(c).to(device)-c)*UU # [numV, 3, 3]
    return R


def Relative_force(pnts, joints, jnorms):
    '''
        pnts: [N, 3],
        joints: [k, 3],
        jnorms: [k, 3]
        output  [N, k], [N, k*3]
    '''
    k = joints.size()[0]
    px = pnts.unsqueeze(1)
    px = px.repeat(1, k, 1)  # [N, k, 3]
    jy = joints.unsqueeze(0)  # [1, k, 3]
    px = px - jy  # [N, k, 3]

    gpx = px.unsqueeze(-1)  # [N, k, 3, 1]
    bnx = jnorms.unsqueeze(1)  # [k, 1, 3]
    sx = torch.matmul(bnx, gpx).squeeze(-1)  # [N, k, 1]
    sx = torch.relu(-sx) # [N, k, 1]
    gf = jnorms.unsqueeze(0)
    gf = sx * gf  # [N, k, 3]

    ax = torch.norm(px, dim=-1)  # [N, k]
    #gf = torch.flatten(gf, 1, 2)  # [N, k*3]

    return ax, sx.squeeze(-1), gf


def Kernel_Dist(dis, sigma):
    '''
    :param dis: [b, N, k]
    :param sigma: scale
    :return:
    '''
    kd = -(dis**2)/(2.*sigma**2)
    kd = torch.exp(kd)
    return kd


def sigmaForce(gf, dis, sigma):
    kd = Kernel_Dist(dis, sigma).unsqueeze(-1)  # [N, k, 1]
    gf = kd * gf
    gf = torch.flatten(gf, 1, 2)
    return gf
