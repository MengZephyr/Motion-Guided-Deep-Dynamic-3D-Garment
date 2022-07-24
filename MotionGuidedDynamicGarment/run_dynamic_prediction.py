from __future__ import division
from __future__ import print_function

import time
import torch
from random import shuffle
from Dynamic_rollout import Dynamic_rollout
from DataIO import load_rtvnFile, npyLoading_pos_norm_vel_acc, save_obj, readMesh_vert_norm, readB2OMap
import os
from geometry import normalize_vetors
import numpy as np

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

modelArch = Dynamic_rollout(GAxisFiles=GAxisFiles, G0FName=G0FName, G_ccName=GVertCCMap, GUVName=GUVName,
                            GK=GK, GkimgH=GkimgS, GkimgW=GkimgS,
                            GMapName=GMapName, GVertPSampleName=GVertPSampleName,
                            BSdfckp=BSDF_ckpName, BSampName=BSampName, B0FName=B0FName, bsize=bsize, device=device0,
                            staticCKP='../ckp_VAE/D_GS1/tshirt/it_50000_vae64_good.ckp',
                            dynamicCKP='../ckp_VAE/D_GD1/tshirt/it_50000_Temp_dyn.ckp')

ifShowRadiuse = True
if ifShowRadiuse:
    radius = modelArch.getSkinningRadius().cpu().numpy()
    print(radius)

caseTest = '/unseen_test/sexy_8/'
frame0 = 2
frame1 = 200

ccMap = readB2OMap(fileRoot + garmUVRoot + "ind_B2O_10.txt")


def getBodyInfo(fID, caseID):
    RTFBodySName = '/body_RTVN_581/'
    fName = fileRoot + caseID + RTFBodySName + str(fID).zfill(6) + '.npy'
    BR, BT, BSverts, BSnorms = load_rtvnFile(fName, device0)

    return BR, BT, BSverts, BSnorms


def getGarmentInfo(fID, caseID):
    garmentFName = '/TShirt_pnva/'
    fName = fileRoot + caseID + garmentFName + str(fID).zfill(6) + '.npy'
    gposit, gnorms, gveloc, gaccle = npyLoading_pos_norm_vel_acc(fName, device0)

    return gposit, gnorms, gveloc, gaccle


def getBodyVertsNormals(fID, caseID):
    BodyNName = '/body_n/'
    fName = fileRoot + caseID + BodyNName + str(fID).zfill(6) + '.obj'
    verts, normals = readMesh_vert_norm(fName, device0)
    normals = normalize_vetors(normals)
    return verts, normals


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


saveRoot = '../rst/test_cat/'

pre_gpose, pre_gnorm, pre_gvelo, pre_gacc = getGarmentInfo(frame0-1, caseTest)
pre_BR, pre_BT, pre_BSVerts, pre_BSNorms = getBodyInfo(frame0-1, caseTest)
BodyVerts, BodyNormals = getBodyVertsNormals(frame0-1, caseTest)
neiList = modelArch.BodyTOGarmenNearest(pre_gpose, BodyVerts)
RefsdfVal = modelArch.calcNearestSDF(pre_gpose, BodyVerts, BodyNormals, neiList).item()

print("Expect sdfVal: ", RefsdfVal)
sdfThre = 5.e-6

ifResetIni = False
if ifResetIni:
    refCaseRoot = '/'
    new_BR, new_BT, new_BSVerts, new_BSNorms = getBodyInfo(frame0 - 1, refCaseRoot)
    new_BodyVerts, new_BodyNormals = getBodyVertsNormals(frame0 - 1, refCaseRoot)

    geo, sdfVal = modelArch.resetIniFrameGarment(gpos=pre_gpose, bSeed=pre_BSVerts,
                                                 new_bVerts=new_BodyVerts, new_bNorms=new_BodyNormals,
                                                 new_bSRotate=new_BR, new_bSTranslate=new_BT, sdfThr=sdfThre,
                                                 maxItter=50)

    save_obj(saveRoot + str(frame0 - 1) + '.obj',
             vertices=recalcVerts(geo[0, :, 0:3].detach().cpu().numpy(), ccMap), faces=None)

    pre_gpose = geo[0, :, :]
    pre_BSVerts = new_BSVerts


for fID in range(frame0, frame1 + 1):
    BR, BT, BSVerts, BSNorms = getBodyInfo(fID, caseTest)
    BodyVerts, BodyNormals = getBodyVertsNormals(fID, caseTest)

    geo, DD, sdfVal = modelArch.OneRoll_run_1(pre_gpos=pre_gpose, pre_gvelo=pre_gvelo, pre_gacc=pre_gacc,
                                              pre_bSeed=pre_BSVerts,
                                              curBVerts=BodyVerts, curBNorms=BodyNormals,
                                              cur_bSeed=BSVerts, curr_bSNorm=BSNorms,
                                              curB_rotate=BR, curB_translate=BT, sdfThr=sdfThre, maxItter=50,
                                              ifpropagate=True)
    print(fID, '--> ', sdfVal)
    save_obj(saveRoot + str(fID) + '.obj', vertices=recalcVerts(geo[0, :, 0:3].detach().cpu().numpy(), ccMap),
             faces=None)
    # save_obj(saveRoot + 'DD/' + str(fID) + '.obj', vertices=recalcVerts(DD[0, :, 0:3].detach().cpu().numpy(), ccMap),
    #          faces=None)

    pre_gacc = (geo[0, :, :] - pre_gpose) - pre_gvelo
    pre_gvelo = geo[0, :, :] - pre_gpose
    pre_gpose = geo[0, :, :]
    pre_BSVerts = BSVerts