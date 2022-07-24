from __future__ import division
from __future__ import print_function

import torch
from Displace_model import NeuralFeatMap,SampleMap
import numpy as np
from DataIO import ReadSampleMap, readMesh_vert_norm_face, readvertPixelMap, writePlyV_F_N_C
from torchvision.utils import save_image

USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device = torch.device("cuda:1" if USE_CUDA else "cpu")

fileRoot = '../data_walk/simple_cases/'
mapName = fileRoot + 'cloth_uv/garment_uvMap_128.txt'

verts, norms, faceID = readMesh_vert_norm_face(fileRoot + 'cloth_uv/garment_geo.obj', device)
numVerts = verts.size()[0]
print(verts.size())
print(norms.size())


levelH, levelW, levelPixelValidX, levelPixelValidY, levelMap, colorV = \
    ReadSampleMap(fileName=mapName, numV=numVerts, outNumLevel=1, ifColor=False)

print(levelMap[0].size())

op_mapping = NeuralFeatMap(device)
normMap = op_mapping(H=levelH[0], W=levelW[0], pX=levelPixelValidX[0], pY=levelPixelValidY[0],
                     vertMask=levelMap[0].type(torch.FloatTensor).to(device), hatF=torch.transpose(norms, 0, 1))
print(normMap.size())

normMap = normMap*0.5 + 0.5
save_image(tensor=normMap, fp='../test/i_2.png')

GSampling = torch.nn.UpsamplingBilinear2d(scale_factor=2).to(device)

vertPixelIndex, vertPixelCoeff = readvertPixelMap(fileRoot + 'cloth_uv/garment_256_vertPixel.txt', numVerts, device)
print(vertPixelIndex.size(), vertPixelCoeff.size())

normMap = GSampling(normMap.unsqueeze(0)).squeeze(0)
print(normMap.size())
normMap = normMap.permute(1, 2, 0)
print(normMap.size())

sampColor = SampleMap(normMap, vertPixelIndex, vertPixelCoeff)
print(sampColor.size())
writePlyV_F_N_C(pDir='../test/s_2.ply', verts=verts.cpu().numpy(), faces=faceID, colors=255. * sampColor.cpu().numpy())

