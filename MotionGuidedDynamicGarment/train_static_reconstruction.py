from __future__ import division
from __future__ import print_function

import time
import torch
from random import shuffle
from StaticPred_architecture import Statics_PredArchi
from DataIO import writePlyV_F_N_C
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.cuda.device_count())

USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device0 = torch.device("cuda:1" if USE_CUDA else "cpu")
device1 = torch.device("cuda:0" if USE_CUDA else "cpu")

fileRoot = '../data_walk/walk_75/'

garmUVRoot = '/uv_bodysuit/'
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
BSampName = fileRoot + bodyUVRoot + 'body_t_581_sample_fArmDLeg.txt'
BSDF_ckpName = fileRoot + bodyUVRoot + '100000_sdf_512.ckp'

modelArch = Statics_PredArchi(GAxisFiles=GAxisFiles, G0FName=G0FName, G_ccName=GVertCCMap,
                              GUVName=GUVName, GK=GK, GkimgH=GkimgS, GkimgW=GkimgS,
                              GMapName=GMapName, GVertPSampleName=GVertPSampleName,
                              B0FName=B0FName, BSdfckp=BSDF_ckpName, BSampName=BSampName, bsize=bsize, device=device0)

modelArch.createNetwork(ckpName=None)
modelArch.createOptimzer()

radius = modelArch.getSkinningRadius().cpu().numpy()
print(radius)

caseTrain = '/'
caseTest = '/unseen_test/walk_90/'

frame0 = 1
frame1 = 300
minframe = 2
fID = [i for i in range(frame0, frame1+1)]
TrainList = fID
TestList = [i for i in range(3, 203)]
Train_KK = len(TrainList) // bsize
Test_KK = len(TestList) // bsize


def getBatchFileList(idList, caseID):
    garmentFName = '/Bodysuit_pnva/'
    RTFName = '/body_RTVN_581/'
    GarmFiles = []
    BRTFiles = []

    for p in idList:
        GarmFiles.append(fileRoot + caseID + garmentFName + str(p).zfill(6) + '.npy')
        BRTFiles.append(fileRoot + caseID + RTFName + str(p).zfill(6) + '.npy')

    return GarmFiles, BRTFiles


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


IFSumWriter = False
if IFSumWriter:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
betit = -1
iterations = 200000

lrFreq = 15000
print('lrFreq: ', lrFreq)
modelArch.setAdjLR_Freq(lrFreq)
for itt in range(betit+1, iterations+1):
    t = time.time()
    k = itt % Train_KK
    if k == 0:
        shuffle(TrainList)
    idList = TrainList[k * bsize: (k + 1) * bsize]
    GarmFiles, BRTFiles = getBatchFileList(idList, caseTrain)
    obj_verts, rand_verts, Loss, vert_objloss, gaussLoss = \
        modelArch.iter_trainNetwork(GarmFiles=GarmFiles, BodyRTfs=BRTFiles, itt=itt)

    print('Iter_{} --> Loss: {:.4f}, LossVert: {:.4f}, gaussLoss: {:.4f}'.
          format(itt, Loss.item(), vert_objloss.item(), gaussLoss.item()))

    if itt % 1000 == 0:
        ckpName = '../ckp_VAE/temp_1000'
        if itt % 5000 == 0:
            ckpName = '../ckp_VAE/D_GS1/temp/it_' + str(itt)
            saveBatchRst('../test_B/t_', idList, itt, obj_verts, None)
            if rand_verts is not None:
                saveBatchRst('../test_B/r_', idList, itt, rand_verts, None)
        modelArch.save_ckp(ckpName, itt)

    if itt % 500 == 0:
        k = (itt // 500) % Test_KK
        if k == 0:
            shuffle(TestList)
        eidList = TestList[k * bsize: (k + 1) * bsize]
        GarmFiles, BRTFiles = getBatchFileList(eidList, caseTest)
        e_obj_verts, e_vert_objloss = \
            modelArch.iter_evalNetwork(GarmFiles=GarmFiles, BodyRTfs=BRTFiles)

        print('Iter_{} --> eval: LossVert: {:.4f}'.format(itt, e_vert_objloss.item()))
        if IFSumWriter:
            writer.add_scalar('Total_Loss', Loss, itt)
            writer.add_scalar('Verts_Loss', vert_objloss, itt)
            writer.add_scalar('eval_recL', e_vert_objloss, itt)

        if itt % 5000 == 0:
            saveBatchRst('../test_B/e_', eidList, itt, e_obj_verts, None)











