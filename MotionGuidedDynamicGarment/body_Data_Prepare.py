from DataIO import *
from geometry import *


def PrepareCanoDeform(GRoot, SaveRoot, sampleIDs, sampleUVs, numV, C_verts, C_norms, frame0, frame1, device):
    prev_A = C_norms.clone()
    R = torch.eye(3).unsqueeze(0)
    R = R.repeat(numV, 1, 1).to(device)  # [numV, 3, 3]
    loss = torch.nn.L1Loss()

    for fID in range(frame0, frame1+1):
        fameB = GRoot + str(fID).zfill(6) + '.npy'
        B_verts, B_norms = npyLoading_vert_norm(fameB, device)
        bb_verts, bb_norms = samplePnts(sampleIDs, sampleUVs, B_verts, B_norms)
        bb_norms = normalize_vetors(bb_norms)


        delt_R = Rotation_A_to_B(prev_A, bb_norms, numV, device)
        R = torch.matmul(delt_R, R)
        prev_A = torch.matmul(R, C_norms.unsqueeze(-1)).squeeze(-1)
        #prev_A = normalize_vetors(prev_A)
        dist = loss(prev_A, bb_norms)
        print('frame:', fID, '-->dist: ', dist.item())

        T = bb_verts - C_verts  # [numV, 3]
        # print(R[0, :, :], T[0, :])
        with open(SaveRoot + str(fID).zfill(6) + '.npy', 'wb') as f:
            np.save(f, R.detach().cpu().numpy())
            np.save(f, T.detach().cpu().numpy())
        exit(1)


def PrepareCanoDeform_directT(GRoot, SaveRoot, sampleIDs, sampleUVs, numV, C_verts, C_norms, frame0, frame1, device):
    #prev_A = C_norms.clone()
    #R = torch.eye(3).unsqueeze(0)
    #R = R.repeat(numV, 1, 1).to(device)  # [numV, 3, 3]
    loss = torch.nn.L1Loss()

    for fID in range(frame0, frame1+1):
        #fameB = GRoot + str(fID).zfill(6) + '.npy'
        #B_verts, B_norms = npyLoading_vert_norm(fameB, device)

        fameB = GRoot + str(fID).zfill(6) + '.obj'
        B_verts, B_norms = readMesh_vert_norm(fameB, device)

        bb_verts, bb_norms = samplePnts(sampleIDs, sampleUVs, B_verts, B_norms)
        bb_norms = normalize_vetors(bb_norms)

        delt_R = Rotation_A_to_B(C_norms, bb_norms, numV, device)
        #R = torch.matmul(delt_R, R)
        prev_A = torch.matmul(delt_R, C_norms.unsqueeze(-1)).squeeze(-1)
        #prev_A = normalize_vetors(prev_A)
        dist = loss(prev_A, bb_norms)
        print('frame:', fID, '-->dist: ', dist.item())

        T = bb_verts - C_verts  # [numV, 3]
        # print(R[0, :, :], T[0, :])
        with open(SaveRoot + str(fID).zfill(6) + '.npy', 'wb') as f:
            np.save(f, delt_R.detach().cpu().numpy())
            np.save(f, T.detach().cpu().numpy())
            np.save(f, bb_verts.detach().cpu().numpy())
            np.save(f, bb_norms.detach().cpu().numpy())


if __name__ == '__main__':
    USE_CUDA = True
    device = torch.device("cuda:2" if USE_CUDA else "cpu")

    GuideRoot = '../data_walk/walk_75/'
    frameC = GuideRoot + '/uv_body/body_geo.obj'
    C_verts, C_norms, _ = readMesh_vert_norm_face(frameC, device)
    print(C_verts.size(), C_norms.size())
    sampleIDs, sampleUVs, sampleLabels = readSamepleInfo(GuideRoot + '/uv_body/body_t_581_sample.txt', True)

    a_verts, a_norms = samplePnts(sampleIDs, sampleUVs, C_verts, C_norms)
    a_norms = normalize_vetors(a_norms)
    print(a_verts.size())
    numV = a_verts.shape[0]
    writePlyV_F_N_C('../test/t_0.ply', a_verts.detach().cpu().numpy(), a_norms.detach().cpu().numpy())

    Frame0 = 1
    Frame1 = 230
    testRoot = GuideRoot + '/unseen_test/rumba_swing/'
    PrepareCanoDeform_directT(testRoot + '/body_n/', testRoot + '/body_RTVN_581/',
                              sampleIDs, sampleUVs,
                              numV, a_verts, a_norms,
                              Frame0, Frame1, device)
