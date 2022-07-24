from __future__ import division
from __future__ import print_function

import time
import torch
from random import shuffle
from DynamicPred_architecture import Dynamics_PredArchi
from DataIO import writePlyV_F_N_C
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.cuda.device_count())

USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device0 = torch.device("cuda:0" if USE_CUDA else "cpu")
device1 = torch.device("cuda:1" if USE_CUDA else "cpu")

fileRoot = '../data_walk/mixamo_shape/'

garmUVRoot = '/skirt_uv/'
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

bodyUVRoot = '/dress_uv/body_uv/'
B0FName = fileRoot + bodyUVRoot + 'body_geo.obj'
BSampName = fileRoot + bodyUVRoot + 'body_t_285_sample.txt'
BSDF_ckpName = fileRoot + bodyUVRoot + '100000_sdf_512.ckp'

modelArch = Dynamics_PredArchi(GAxisFiles=GAxisFiles, G0FName=G0FName, G_ccName=GVertCCMap, GUVName=GUVName,
                               GK=GK, GkimgH=GkimgS, GkimgW=GkimgS,
                               GMapName=GMapName, GVertPSampleName=GVertPSampleName,
                               BSdfckp=BSDF_ckpName, BSampName=BSampName, B0FName=B0FName, bsize=bsize, device=device0)

staticCkpRoot = '../ckp_VAE/D_GS1/mixamo_2Layer_skirt/'
modelArch.createDynamicNetwork(static_ckp=staticCkpRoot + 'it_120000_vae64_z.ckp',
                               dyn_ckp=None)
modelArch.createDynamicOptimizer()

print(modelArch.skinningKernelRadiuse.detach().cpu().numpy())

caseTrain = '/walk_75/'
caseTest = '/walk_75/'


with open(staticCkpRoot + 'Z_Static.npy', 'rb') as f:
    ZID0 = np.load(f)
    Train_Z = np.load(f)
Train_Z = torch.from_numpy(Train_Z).type(torch.FloatTensor).to(device0)
print(ZID0, Train_Z.size())

frame0 = ZID0+1
frame1 = 300
minframe = 2
fID = [i for i in range(frame0, frame1+1)]
TrainList = fID
TestList = [i for i in range(3, 203)]
Train_KK = len(TrainList) // bsize
Test_KK = len(TestList) // bsize


def getBatchFileList(idList, caseID, ZZArray):
    garmentFName = '/Skirt_pnva/'
    RTFBodySName = '/dressBody_RTVN_285/'

    curGarmFiles = []
    curBodySFiles = []

    preGarmFiles = []
    preBodySFiles = []

    for p in idList:
        curGarmFiles.append(fileRoot + caseID + garmentFName + str(p).zfill(6) + '.npy')
        curBodySFiles.append(fileRoot + caseID + RTFBodySName + str(p).zfill(6) + '.npy')

        preGarmFiles.append(fileRoot + caseID + garmentFName + str(p-1).zfill(6) + '.npy')
        preBodySFiles.append(fileRoot + caseID + RTFBodySName + str(p-1).zfill(6) + '.npy')

    if ZZArray is not None:
        preGZ = ZZArray[[i-1-ZID0 for i in idList], :]
        currGZ = ZZArray[[i - ZID0 for i in idList], :]
    else:
        preGZ = None
        currGZ = None

    return preGarmFiles, preBodySFiles, preGZ, curGarmFiles, curBodySFiles, currGZ


def saveBatchRst(proot, idList, it, rst, rstN):
    numS = 1
    for i in range(numS):
        fID = idList[i]
        rstv = rst[i, :, :].detach().cpu().numpy()
        if rstN is not None:
            nn = rstN[i, :, :].detach().cpu().numpy()
            writePlyV_F_N_C(proot + str(it) + '_' + str(fID) + '.ply', verts=rstv[:, 0:3], normals=nn,
                            colors=None, faces=None)
        else:
            writePlyV_F_N_C(proot + str(it) + '_' + str(fID) + '.ply', verts=rstv[:, 0:3], normals=None,
                            colors=None, faces=None)


IFSumWriter = True
if IFSumWriter:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
betit = -1
iterations = 100000

adjSeq = 10000
print(adjSeq)
modelArch.setAdjLR_Freq(adjSeq)

for itt in range(betit+1, iterations+1):
    t = time.time()
    k = itt % Train_KK
    if k == 0:
        shuffle(TrainList)
    idList = TrainList[k * bsize: (k + 1) * bsize]
    preGarmFiles, preBodySFiles, preGZ, curGarmFiles, curBodySFiles, currGZ = \
        getBatchFileList(idList, caseTrain, Train_Z)

    pre_geo, cur_geo, Loss, geo_loss, z_loss, gaussLoss = \
        modelArch.Iter_trainDynNetwork(preGarmFiles=preGarmFiles, preBodySeedFiles=preBodySFiles, preGZ=preGZ,
                                       curGarmentFiles=curGarmFiles, curBodySeedFiles=curBodySFiles, currGZ=currGZ,
                                       itt=itt)

    print('Iter_{} --> Loss: {:.4f}, LossVert: {:.4f}, z_loss: {:.4f} '.
          format(itt, Loss.item(), geo_loss.item(), z_loss.item()))

    if itt % 1000 == 0:
        ckpName = '../ckp_VAE/tempC1_1000'
        if itt % 5000 == 0:
            ckpName = '../ckp_VAE/D_GD1/mixamo_2Layer_skirt/it_' + str(itt)
            saveBatchRst('../test_B/t_', idList, itt, cur_geo, None)
            saveBatchRst('../test_B/t_', [i-1 for i in idList], itt, pre_geo, None)
        modelArch.save_dynckp_d(ckpName, itt)

    if itt % 500 == 0:
        k = (itt // 500) % Test_KK
        if k == 0:
            shuffle(TestList)
        eidList = TestList[k * bsize: (k + 1) * bsize]
        tpreGarmFiles, tpreBodySFiles, tpreGZ, tcurGarmFiles, tcurBodySFiles, tcurrGZ = \
            getBatchFileList(eidList, caseTest, None)
        tpre_geo, t_geo, t_vert_loss = \
            modelArch.Iter_evalDynNetwork(preGarmFiles=tpreGarmFiles, preBodySeedFiles=tpreBodySFiles,
                                          curGarmentFiles=tcurGarmFiles, curBodySeedFiles=tcurBodySFiles)

        print('Iter_{} --> eval: LossVert: {:.4f}'.format(itt, t_vert_loss.item()))

        if itt % 5000 == 0:
            saveBatchRst('../test_B/e_', eidList, itt, t_geo, None)
            saveBatchRst('../test_B/e_', [i - 1 for i in eidList], itt, tpre_geo, None)

        if itt % 500 == 0:
            if IFSumWriter:
                writer.add_scalar('Total_Loss', Loss, itt)
                writer.add_scalar('Verts_Loss', geo_loss, itt)
                writer.add_scalar('eval_recL', t_vert_loss, itt)


