import os
import torch
import numpy as np
from geometry import normalize_vetors


def readSamepleInfo(fname, ifFlag=True):
    if not (os.path.exists(fname)):
        return None, None
    vIDs = []
    UVs = []
    Fs = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if ifFlag:
            ff = int(values[0])
            ids = [int(x) for x in values[1:4]]
            uv = [float(x) for x in values[4:6]]
            vIDs.append(ids)
            UVs.append(uv)
            Fs.append(ff)
        else:
            ids = [int(x) for x in values[0:3]]
            uv = [float(x) for x in values[3:5]]
            vIDs.append(ids)
            UVs.append(uv)
    return vIDs, UVs, Fs


def readFileLine(file):
    line = file.readline()
    values = line.split()
    return values


def ReadSampleMap(fileName, numV, outNumLevel=5, ifColor=False):
    file = open(fileName, "r")
    if ifColor is True:
        colorV = [float(x) for x in readFileLine(file)]
    else:
        colorV = [-1., -1., -1.]
    numLevel = int(readFileLine(file)[0])
    levelH = [int(x) for x in readFileLine(file)]
    levelW = [int(x) for x in readFileLine(file)]
    levelPixelValidX = []
    levelPixelValidY = []
    levelMap = []
    lc = 0
    for le in range(numLevel):
        if lc >= outNumLevel:
            break

        levelInfo = readFileLine(file)
        levelID = int(levelInfo[0])
        numValid = int(levelInfo[1])
        PixelValidX = []
        PixelValidY = []
        Map = torch.zeros(numV, numValid)
        for vi in range(numValid):
            values = readFileLine(file)
            PixelValidX.append(int(values[0]))
            PixelValidY.append(int(values[1]))
            ind = [int(values[2]), int(values[3]), int(values[4])]
            u = float(values[5])
            v = float(values[6])
            k = torch.tensor([1.-u-v, u, v])
            Map[ind, vi] = k
        levelMap.append(Map)
        levelPixelValidX.append(PixelValidX)
        levelPixelValidY.append(PixelValidY)
        lc += 1

    file.close()

    return levelH, levelW, levelPixelValidX, levelPixelValidY, levelMap, colorV


def readvertPixelMap(fileName, vertnum, device):
    file = open(fileName, "r")
    vertPixelIndex = []
    vertPixelCoeff = []
    values = readFileLine(file)
    numV = int(values[1])
    assert (numV == vertnum)
    for vi in range(numV):
        pcorner = []
        values = readFileLine(file)
        pcorner.append([int(values[0]), int(values[1])])
        pcorner.append([int(values[2]), int(values[3])])
        pcorner.append([int(values[4]), int(values[5])])
        pcorner.append([int(values[6]), int(values[7])])
        vertPixelCoeff.append([float(values[8]), float(values[9]), float(values[10]), float(values[11])])
        vertPixelIndex.append(pcorner)

    vertPixelIndex = torch.tensor(vertPixelIndex).to(device)
    vertPixelCoeff = torch.tensor(vertPixelCoeff).to(device)
    return vertPixelIndex, vertPixelCoeff


def readMesh_vert_norm(fname, device):
    if not (os.path.exists(fname)):
        print('No file.')
        return None, None
    posArray = []
    normArray = []

    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            posArray.append([v[0], v[1], v[2]])
        if values[0] == 'vn':
            v = [float(x) for x in values[1:4]]
            normArray.append([v[0], v[1], v[2]])

    verts = np.array(posArray)
    norms = np.array(normArray)

    if device is not None:
        verts = torch.from_numpy(verts).type(torch.FloatTensor).to(device)
        norms = torch.from_numpy(norms).type(torch.FloatTensor).to(device)

    return verts, norms


def readMesh_vert_norm_face(fname, device):
    if not (os.path.exists(fname)):
        print('No file.')
        return None, None, None
    posArray = []
    normArray = []
    faceArray = []

    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            posArray.append([v[0], v[1], v[2]])
        if values[0] == 'vn':
            v = [float(x) for x in values[1:4]]
            normArray.append([v[0], v[1], v[2]])
        if values[0] == 'f':
            f = [int(x.split('/')[0]) - 1 for x in values[1:4]]
            faceArray.append(f)

    verts = np.array(posArray)
    norms = np.array(normArray)
    faceArray = np.array(faceArray)

    if device is not None:
        verts = torch.from_numpy(verts).type(torch.FloatTensor).to(device)
        norms = torch.from_numpy(norms).type(torch.FloatTensor).to(device)

    return verts, norms, faceArray


def readAxisFile(name, device=None):
    array = []
    file = open(name, "r")
    for line in file:
        values = line.split()
        if len(values) > 1:
            v = [float(x) for x in values]
            array.append([v[0], v[1], v[2]])
    file.close()
    array = np.array(array)

    if device is not None:
        array = torch.from_numpy(array).type(torch.FloatTensor).to(device)

    return array


def npyLoading_vert_norm(fname, device):
    with open(fname, 'rb') as f:
        verts = np.load(f)
        norms = np.load(f)
    verts = torch.from_numpy(verts).type(torch.FloatTensor).to(device)
    norms = torch.from_numpy(norms).type(torch.FloatTensor).to(device)
    norms = normalize_vetors(norms)
    return verts, norms


def npyLoading_pos_norm_vel_acc(fname, device):
    with open(fname, 'rb') as f:
        posit = np.load(f)
        norms = np.load(f)
        veloc = np.load(f)
        accle = np.load(f)

    posit = torch.from_numpy(posit).type(torch.FloatTensor).to(device)
    norms = torch.from_numpy(norms).type(torch.FloatTensor).to(device)
    veloc = torch.from_numpy(veloc).type(torch.FloatTensor).to(device)
    accle = torch.from_numpy(accle).type(torch.FloatTensor).to(device)
    norms = normalize_vetors(norms)

    return posit, norms, veloc, accle


def readB2OMap(fName):
    if not (os.path.exists(fName)):
        return None
    mapArray = []
    file = open(fName, "r")
    numBV = 0
    for line in file:
        values = line.split()
        if values[0] == '#':
            numBV = int(values[1])
        else:
            mm = [int(x) for x in values[1:int(values[0])+1]]
            mapArray.append(mm)
    if not (len(mapArray) == numBV):
        print("Error:: Data wrong in mapping file.")
        return None
    return mapArray


def writePlyV_F_N_C(pDir, verts, normals=None, colors=None, faces=None):
    numVerts = verts.shape[0]
    if faces is not None:
        numFace = faces.shape[0]
    with open(pDir, 'w') as f:
        f.write("ply\n" + "format ascii 1.0\n")
        f.write("element vertex " + str(numVerts) + "\n")
        f.write("property float x\n" + "property float y\n" + "property float z\n")
        if normals is not None:
            f.write("property float nx\n" + "property float ny\n" + "property float nz\n")
        if colors is not None:
            f.write("property uchar red\n" + "property uchar green\n"
                    + "property uchar blue\n" + "property uchar alpha\n")
        if faces is not None:
            f.write("element face " + str(numFace) + "\n")
            f.write("property list uchar int vertex_indices\n")

        f.write("end_header\n")
        for p in range(numVerts):
            if normals is not None and colors is not None:
                v = verts[p]
                c = colors[p]
                n = normals[p]
                f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                        + str(n[0]) + " " + str(n[1]) + " " + str(n[2]) + " "
                        + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + "255\n")
            elif normals is not None:
                v = verts[p]
                n = normals[p]
                f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                        + str(n[0]) + " " + str(n[1]) + " " + str(n[2]) + "\n")
            elif colors is not None:
                v = verts[p]
                c = colors[p]
                f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                        + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + "255\n")
            else:
                v = verts[p]
                f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")

        if faces is not None:
            for p in range(numFace):
                fds = faces[p]
                f.write("3 " + str(fds[0]) + " " + str(fds[1])
                        + " " + str(fds[2]) + "\n")
        f.close()


def save_obj(filename, vertices, faces=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if faces is not None:
            for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def load_rtFile(fname, device):
    with open(fname, 'rb') as f:
        R = np.load(f)
        T = np.load(f)
        Sv = np.load(f)
    R = torch.from_numpy(R).type(torch.FloatTensor).to(device)
    T = torch.from_numpy(T).type(torch.FloatTensor).to(device)
    Sv = torch.from_numpy(Sv).type(torch.FloatTensor).to(device)
    return R, T, Sv


def load_rtvnFile(fname, device):
    with open(fname, 'rb') as f:
        R = np.load(f)
        T = np.load(f)
        Sv = np.load(f)
        Sn = np.load(f)
    R = torch.from_numpy(R).type(torch.FloatTensor).to(device)
    T = torch.from_numpy(T).type(torch.FloatTensor).to(device)
    Sv = torch.from_numpy(Sv).type(torch.FloatTensor).to(device)
    Sn = torch.from_numpy(Sn).type(torch.FloatTensor).to(device)
    return R, T, Sv, Sn
