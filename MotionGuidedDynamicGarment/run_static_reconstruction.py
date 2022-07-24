from __future__ import division
from __future__ import print_function

import time
import torch
from random import shuffle
from StaticPred_architecture import Statics_PredArchi
from DataIO import save_obj, readB2OMap
import os
import numpy as np
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.cuda.device_count())

USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device0 = torch.device("cuda:0" if USE_CUDA else "cpu")
device1 = torch.device("cuda:1" if USE_CUDA else "cpu")

fileRoot = '../data_walk/walk_75/'

garmUVRoot = '/uv_tshirt/'
GAxisFiles = {'x': fileRoot + garmUVRoot + 'garment_canc_tangents.txt',
              'y': fileRoot + garmUVRoot + 'garment_canc_basements.txt',
              'z': fileRoot + garmUVRoot + 'garment_canc_normals.txt'}

G0FName = fileRoot + garmUVRoot + 'garment_geo.obj'
GVertCCMap = fileRoot + garmUVRoot + 'ind_B2O_10.txt'
GUVName = fileRoot + garmUVRoot + 'garment_uv.obj'
GMapName = fileRoot + garmUVRoot + 'garment_uvMap_128.txt'
GVertPSampleName = fileRoot + garmUVRoot + 'garment_256_vertPixel.txt'

GK = 8
GkimgS = 256
bsize = 1

bodyUVRoot = '/uv_body/'
B0FName = fileRoot + bodyUVRoot + 'body_geo.obj'
BSampName = fileRoot + bodyUVRoot + 'body_t_581_sample.txt'
BSDF_ckpName = fileRoot + bodyUVRoot + '100000_sdf_512.ckp'

modelArch = Statics_PredArchi(GAxisFiles=GAxisFiles, G0FName=G0FName, G_ccName=GVertCCMap,
                              GUVName=GUVName, GK=GK, GkimgH=GkimgS, GkimgW=GkimgS,
                              GMapName=GMapName, GVertPSampleName=GVertPSampleName,
                              B0FName=B0FName, BSdfckp=BSDF_ckpName, BSampName=BSampName, bsize=bsize, device=device0)

modelArch.createNetwork(ckpName='../ckp_VAE/D_GS1/tshirt/it_50000_vae64_good.ckp')

ifShowRadiuse = True
if ifShowRadiuse:
    radius = modelArch.getSkinningRadius().cpu().numpy()
    print(radius)

caseTest = '/'
frame0 = 1
frame1 = 200

ccMap = readB2OMap(fileRoot + garmUVRoot + "ind_B2O_10.txt")


def recalcVerts(verts, ccMap, vTans=[0, 0, 0]):
    nverts = np.zeros_like(verts)
    for cc in ccMap:
        if len(cc) > 1:
            vv = np.sum(verts[cc], axis=0)
            vv = vv/float(len(cc))
            verts[cc] = vv
        nverts[cc, 0] = verts[cc, 0] + vTans[0]
        nverts[cc, 1] = verts[cc, 1] + vTans[1]
        nverts[cc, 2] = verts[cc, 2] + vTans[2]
    return nverts


def getBatchFileList(idList, caseID):
    garmentFName = '/TShirt_pnva/'
    RTFName = '/body_RTVN_581/'
    GarmFiles = []
    BRTFiles = []

    for p in idList:
        GarmFiles.append(fileRoot + caseID + garmentFName + str(p).zfill(6) + '.npy')
        BRTFiles.append(fileRoot + caseID + RTFName + str(p).zfill(6) + '.npy')

    return GarmFiles, BRTFiles


def saveBatchRst(proot, idList, rst):
    for i in range(len(idList)):
        fID = idList[i]
        rstv=recalcVerts(rst[i, :, 0:3].detach().cpu().numpy(), ccMap)
        #rstv = rst[i, :, :].detach().cpu().numpy()
        save_obj(proot + str(fID) + '.obj', vertices=rstv[:, 0:3], faces=None)

Loss = []
ZLatent = []
ifsaveZ = False
for fID in range(frame0, frame1+1):
    GarmFiles, BRTFiles = getBatchFileList([fID], caseTest)
    DD, geoV, midz, lossgeo = modelArch.run_Network(GarmFiles, BRTFiles)
    ZLatent.append(midz)
    print('frame_', fID, '--> ', lossgeo.item())
    Loss.append(lossgeo.item())
    saveBatchRst('../rst/', [fID], geoV)
    #saveBatchRst('../rst/walk75_twoL_skirt_D/', [fID], DD)

if ifsaveZ:
    ZLatent = torch.cat(ZLatent, dim=0)
    ZLatent = ZLatent.detach().cpu().numpy()
    print(ZLatent.shape)
    with open('../ckp_VAE/Z_Static.npy', 'wb') as f:
        np.save(f, frame0)
        np.save(f, ZLatent)
        f.close()

FX = [i for i in range(frame0, frame1 + 1)]
fig, ax = plt.subplots()
ax.set_title('Loss varying with frames', fontsize=15, fontweight='demi')
ax.set_xlabel('Frame_t', fontsize=12)
ax.set_ylabel('loss', fontsize=12)
ax.plot(FX, Loss, marker='.')
plt.savefig('../rst/loss.svg')
plt.close(fig)

Loss = np.array(Loss)
print(np.mean(Loss), np.std(Loss))
