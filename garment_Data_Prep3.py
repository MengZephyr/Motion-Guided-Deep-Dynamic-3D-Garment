import numpy as np
import os

def readMesh_vert_norm(fname, device):
    if not (os.path.exists(fname)):
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
    
def readTextureOBJFile_twoFaces(fname):
    if not(os.path.exists(fname)):
        return None, None, None, None
    vertArray = []
    normArray = []
    vfaceArray = []
    tfaceArray = []
    textArray = []
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
        if values[0] == 'vt':
            vt = [float(x) for x in values[1:3]]
            textArray.append(vt)
        if values[0] == 'f':
            f = [int(x.split('/')[0]) for x in values[1:4]]
            vfaceArray.append(f)
            ft = [int(x.split('/')[1]) for x in values[1:4]]
            tfaceArray.append(ft)
            
    vertArray = np.array(vertArray, dtype=np.float64)
    normArray = np.array(normArray)
    vfaceArray = np.array(vfaceArray)
    textArray = np.array(textArray)
    tfaceArray = np.array(tfaceArray)

    return vertArray, normArray, vfaceArray, textArray, tfaceArray
    

def prepareVertsInfos_VelAcc(froot, saveRoot, frame0, frame1):
    #h = 1.0 / 30.0 # 30 fps
    
    verts, norms = readMesh_vert_norm(froot + str(frame0).zfill(7) + '.obj', None)
    velos = np.zeros_like(verts)

    for i in range(frame0, frame1+1):
        i_verts, i_norms = readMesh_vert_norm(froot + str(i).zfill(7) + '.obj', None)
        i_velos = (i_verts - verts) 
        i_accle = (i_velos - velos)
        with open(saveRoot + str(i).zfill(6) + '.npy', 'wb') as f:
            np.save(f, i_verts)
            np.save(f, i_norms)
            np.save(f, i_velos)
            np.save(f, i_accle)
        verts = i_verts.copy()
        velos = i_velos.copy()


def prepareVertsNormPose(froot, saveRoot, frame0, frame1):
    for i in range(frame0, frame1+1):
        i_verts, i_norms = readMesh_vert_norm(froot + str(i).zfill(6) + '.obj', None)
        with open(saveRoot + str(i).zfill(6) + '.npy', 'wb') as f:
            np.save(f, i_verts)
            np.save(f, i_norms)
            

def savePly(pDir, verts):
    numVerts = verts.shape[0]
    with open(pDir, 'w') as f:
        f.write("ply\n" + "format ascii 1.0\n")
        f.write("element vertex " + str(numVerts) + "\n")
        f.write("property float x\n" + "property float y\n" + "property float z\n")
        f.write("end_header\n")
        for p in range(numVerts):
            v = verts[p]
            f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
        
        f.close()

        
def prepareTranslation(ref_objfile, froot, saveRoot, frame0, frame1):
    ref_verts, _ = readMesh_vert_norm(ref_objfile, None)
    
    translations = []
    for i in range(frame0, frame1+1):
        i_verts, i_norms = readMesh_vert_norm(froot + str(i).zfill(6) + '.obj', None)
        i_translation = i_verts - ref_verts
        #print(i_translation)
        mean_translation = np.mean(i_translation, axis=0)
        #print(mean_translation)
        
        tt = ref_verts + mean_translation
        
        #savePly('test.ply', tt)
        #g_verts, g_norms = readMesh_vert_norm(saveRoot + '/simulation_n/'+ str(i).zfill(7) + '.obj', None)
        #tg = g_verts - mean_translation
        #savePly('test_g.ply', tg)
        #exit(1)
        
    
        loss = np.mean(np.abs(tt-i_verts))
        print(i, '--> ', loss)
        
        translations.append(mean_translation)
    
    translations = np.array(translations)
    print(translations.shape)
    with open(saveRoot + 'translations.npy', 'wb') as f:
        np.save(f, translations)
    


if __name__== '__main__':
    ifGarment = True
    ifBody = False
    ifTrans = False
    if ifGarment:
        GuideRoot = "D:/models/DS/Data_walk/Long_90/"
        prepareVertsInfos_VelAcc(GuideRoot + '/Bodysuit_PD10/PD10_', GuideRoot+'/Bodysuit_pnva/', 1, 1171)
        
    # if ifBody:
        # GuideRoot = "D:/models/simple_case/data/tbending_2/"
        # prepareVertsNormPose(GuideRoot + '/body_n/', GuideRoot+'/stick_pn/', 1, 250)
        
    # if ifTrans:
        # fRoot = "D:/models/simple_case/data/"
        # RefObjF = fRoot + "stick_uv/body_geo.obj"
        # GGRoot = fRoot + "translation_2/stick_n/"
        # prepareTranslation(RefObjF, GGRoot, fRoot+"translation_2/", 1, 210)