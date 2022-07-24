import numpy as np
import os
from sklearn.neighbors import KDTree


def readOBJFile(fname):
    if not(os.path.exists(fname)):
        return None, None, None
    vertArray = []
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
            vertArray.append(v)
        if values[0] == 'vn':
            vn = [float(x) for x in values[1:4]]
            normArray.append(vn)
        if values[0] == 'f':
            f = [int(x.split('/')[0]) for x in values[1:4]]
            faceArray.append(f)
    vertArray = np.array(vertArray, dtype=np.float64)
    normArray = np.array(normArray)
    faceArray = np.array(faceArray)

    return vertArray, normArray, faceArray


def saveO2B(fname, indA_O2B):
    with open(fname, 'w') as f:
        f.write("# " + str(indA_O2B.shape[0]) + "\n")
        for c in range(indA_O2B.shape[0]):
            f.write(str(indA_O2B[c][0]) + "\n")
        f.close()


def saveB2O(fname, indA_B2O):
    cc = 0
    with open(fname, 'w') as f:
        f.write("# " + str(indA_B2O.shape[0]) + "\n")
        for c in range(indA_B2O.shape[0]):
            numP = indA_B2O[c].shape[0]
            if numP > 1:
                cc += 1
            f.write(str(numP))
            for d in range(numP):
                f.write(" " + str(indA_B2O[c][d]))
            f.write("\n")
        f.close()
    print(cc)


def vertArrayMapping(BaseName, OrigName, saveRoot="./"):
    # BaseName = './Base.obj'
    # OrigName = './Orig.obj'

    B_Vert, _, _ = readOBJFile(BaseName)
    O_Vert, _, _ = readOBJFile(OrigName)

    print("O: ", O_Vert.shape, "B: ", B_Vert.shape)

    '''Orig --> Base, i.e. 2 --> 1'''
    B_tree = KDTree(B_Vert)
    dist, ind_O2B = B_tree.query(O_Vert, k=1)
    print(ind_O2B.shape)

    O_tree = KDTree(O_Vert)
    ind_B2O = O_tree.query_radius(B_Vert, r=1.e-6)
    print(ind_B2O.shape)
    #print(ind_B2O)

    saveO2B(saveRoot+"ind_O2B_10.txt", ind_O2B)
    saveB2O(saveRoot+"ind_B2O_10.txt", ind_B2O)


def faceCenterMapping(BaseName, OrigName):
    '''
    Check
    '''
    # BaseName = 'D:/models/MD/DataModel/DressOri/case_1/Base10.obj'
    # OrigName = 'D:/models/MD/DataModel/DressOri/case_1/Ori10.obj'

    B_Vert, _, B_Face = readOBJFile(BaseName)
    print(B_Vert.shape, B_Face.shape)
    O_Vert, _, O_Face = readOBJFile(OrigName)
    print(O_Vert.shape, O_Face.shape)

    B_Center = calcFaceCenter(B_Vert, B_Face)
    O_Center = calcFaceCenter(O_Vert, O_Face)
    print(B_Center.shape, O_Center.shape)
    B_tree = KDTree(B_Center)
    dist, ind_O2B = B_tree.query(O_Center, k=1)
    print(ind_O2B.shape)

    saveO2B("D:/models/MD/DataModel/DressOri/case_1/Face_T2B.txt", ind_O2B)


def calcFaceCenter(vert, face):
    centers = []
    numF = face.shape[0]
    for f in range(numF):
        fId = face[f]
        fId = [fId[0]-1, fId[1]-1, fId[2]-1]
        vc = [0., 0., 0.]
        vc[0] = (vert[fId[0]][0] + vert[fId[1]][0] + vert[fId[2]][0]) / 3.
        vc[1] = (vert[fId[0]][1] + vert[fId[1]][1] + vert[fId[2]][1]) / 3.
        vc[2] = (vert[fId[0]][2] + vert[fId[1]][2] + vert[fId[2]][2]) / 3.
        centers.append(vc)
    return np.array(centers)


if __name__ == '__main__':
    caseName = '/'
    prefRoot = 'D:/models/DS/Data_walk/mixamo_body/sequence/dress_uv/'
    BaseName = prefRoot + caseName + '/base.obj'
    OrigName = prefRoot + caseName + '/dress.obj'
    saveRoot = prefRoot + caseName + '/'
    vertArrayMapping(BaseName=BaseName, OrigName=OrigName, saveRoot=saveRoot)



