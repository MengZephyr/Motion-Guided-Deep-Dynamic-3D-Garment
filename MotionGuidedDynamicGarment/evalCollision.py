from sklearn import neighbors
import numpy as np
from DataIO import *

USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device1 = torch.device("cuda:1" if USE_CUDA else "cpu")

bodyFolder = '../data_walk/walk_75/unseen_test/thin_body_salsa_swing/body_n/'

OptgarmentFolder = '../rst/salsa/'
#OrigarmentFolder = '../rst/rst_75_without/'


def calcCollision(bodyTree, garmverts, body_verts, body_normals):
    dist, bInd = bodyTree.query(garmverts.cpu().numpy(), k=1)
    neiList = [i[0] for i in bInd]
    gpx = garmverts - body_verts[list(neiList), :]
    bnx = body_normals[list(neiList), :]

    gpx = gpx.unsqueeze(-1)
    bnx = bnx.unsqueeze(1)
    sx = torch.matmul(bnx, gpx).squeeze(-1)
    sx = torch.relu(-sx)

    numV = sx.size()[0]

    ccdect = torch.nonzero(sx)
    numC = ccdect.size()[0]

    ratioCC = float(numC) / float(numV)
    return ratioCC

Opt_CCR = []
#Ori_CCR = []

for fID in range(20, 150):
    bodyFile = bodyFolder + str(fID).zfill(6) + '.obj'
    body_verts, body_normals = readMesh_vert_norm(bodyFile, device1)
    bodyTree = neighbors.KDTree(body_verts.cpu().numpy())

    OptgarmentFile =OptgarmentFolder + str(fID) + '.obj'
    Optg_verts, _ = readMesh_vert_norm(OptgarmentFile, device1)
    #OptgarmentFile =OptgarmentFolder + str(fID).zfill(6) + '.npy'
    #Optg_verts, _, _ = load_pnvFile(OptgarmentFile, device1)

    #OrigarmentFile = OrigarmentFolder + str(fID) + '.obj'
    #Orig_verts, _ = readMesh_vert_norm(OrigarmentFile, device1)

    Opt_r = calcCollision(bodyTree, Optg_verts, body_verts, body_normals)
    #Ori_r = calcCollision(bodyTree, Orig_verts, body_verts, body_normals)
    #print('Frame_', fID, ': ori ccr: ', Ori_r, ' Opt handling ccr: ', Opt_r)
    print('Frame_', fID,  ' Opt handling ccr: ', Opt_r)

    Opt_CCR.append(Opt_r)
    #Ori_CCR.append(Ori_r)

#Ori_CCR = np.array(Ori_CCR)
Opt_CCR = np.array(Opt_CCR)

#print('Ori: mean: ', np.mean(Ori_CCR), ' std: ', np.std(Ori_CCR))
print('Opt: mean: ', np.mean(Opt_CCR), ' std: ', np.std(Opt_CCR))







