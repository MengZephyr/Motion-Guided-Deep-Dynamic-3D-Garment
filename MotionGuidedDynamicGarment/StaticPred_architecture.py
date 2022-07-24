from Displace_model import *
from DataIO import *
from geometry import *
from IGR_Net import SDF_Model


class Statics_PredArchi(object):
    def __init__(self, GAxisFiles, G0FName, G_ccName, GUVName, GK, GkimgH, GkimgW, GMapName, GVertPSampleName,
                 B0FName, BSdfckp, BSampName, bsize, device):
        self.device = device
        self.bsize = bsize

        self.ifNeuralTexture = False
        if self.ifNeuralTexture:
            self.pFeatDim = 64
        else:
            self.pFeatDim = 0

        self.featTensor = None

        # garment G0
        self.G0_TVerts, _, G_faceIDs = readMesh_vert_norm_face(G0FName, device)
        self.G_numVs = self.G0_TVerts.size()[0]
        self.G_edgesID = get_edges(G_faceIDs)
        self.G_lapMatrix = getLaplacianMatrix(self.G_edgesID, self.G_numVs).to(self.device)

        if G_ccName is not None:
            ccMap = readB2OMap(G_ccName)
            GCCEdge = getCCEdges(ccMap)
            self.G_edgesID = np.concatenate((self.G_edgesID, GCCEdge), axis=0)

        print(self.G_edgesID.shape)
        self.G_edgeLen = compute_edgeLength(self.G0_TVerts.unsqueeze(0), self.G_edgesID)
        self.G_edgeLen = self.G_edgeLen.repeat(self.bsize, 1)
        print("G edges: ", self.G_edgeLen.size())

        # garment H0
        self.G0_TAxis, self.G0_BAxis, self.G0_NAxis = self.readConanicalAxises(GAxisFiles, self.device)

        print(self.G0_TAxis.size(), self.G0_BAxis.size(), self.G0_NAxis.size(), self.G0_TVerts.size(), self.G_numVs)

        # garment uv
        vertTensor, _, _f = readMesh_vert_norm_face(GUVName, self.device)
        uvTensor = vertTensor[:, 0:2]
        uvTensor[:, 0] = GkimgW * uvTensor[:, 0]
        uvTensor[:, 1] = GkimgH * uvTensor[:, 1]
        self.G_KuvPoseEn = uvPoseEncoding(uvTensor, GK)
        self.G_KuvPoseEn.requires_grad = False
        self.GK = GK
        print(self.G_KuvPoseEn.size())

        # garment vert_2_map
        self.G_levelH, self.G_levelW, self.G_levelPixelValidX, self.G_levelPixelValidY, self.G_levelMap = \
            self.readMapSampleFile(mapName=GMapName, numV=self.G_numVs, device=self.device)

        # garment map_2_vert
        self.G_vertPixelIndex, self.G_vertPixelCoeff = \
            readvertPixelMap(GVertPSampleName, self.G_numVs, self.device)

        # body sdf at pose_0
        self.B_SDFModel = SDF_Model(BSdfckp, self.device)
        self.B_sdfThre = torch.tensor([0.0006]).unsqueeze(0).unsqueeze(0).to(self.device)

        B0Verts, _, _f = readMesh_vert_norm_face(B0FName, self.device)
        self.B_numVs = B0Verts.size()[0]
        self.BBsampleIDs, self.BBsampleUVs, self.BBSampleLabel = readSamepleInfo(BSampName, ifFlag=True)
        self.BB_numSamples = len(self.BBsampleIDs)
        self.B0Joints,  _ = samplePnts(self.BBsampleIDs, self.BBsampleUVs, B0Verts, None)
        self.NumSeedLables = max(self.BBSampleLabel) + 1
        print("Num seed labels: ", self.NumSeedLables)

        # networks
        self.skinningKernelRadiuse = None
        self.poseMapping = NeuralFeatMap(self.device).to(self.device)
        self.RelativePos_Encoder = None
        self.Pred_Encoder = None
        self.Pred_Decoder = None
        self.Optimizer = None

        self.ini_lr = [1.e-4, 1.e-4]
        self.adfacetor = [0.5, 0.5]
        self.adlr_freq = 5000
        self.jSDFSigma = 0.01

        # loss
        self.L1DataLoss = torch.nn.L1Loss().to(self.device)

    def setAdjLR_Freq(self, freq):
        self.adlr_freq = freq

    def getSkinningRadius(self):
        return self.skinningKernelRadiuse.clone().detach()

    def createNetwork(self, ckpName=None):
        if self.ifNeuralTexture:
            self.featTensor = torch.randn(self.pFeatDim, self.G_levelH, self.G_levelW)

        #self.skinningKernelRadiuse = torch.ones(self.BB_numSamples) * 0.1
        self.skinningKernelRadiuse = torch.ones(self.NumSeedLables) * 0.05

        self.RelativePos_Encoder = RelativePosEncoder(in_ch=self.BB_numSamples*3, out_ch=128).to(self.device)
        self.Pred_Encoder = DisEncoder(in_ch=128+self.pFeatDim, out_ch=1024, midSize=64).to(self.device)
        self.Pred_Decoder = Pred_decoder(in_ch=1024, midSize=64, uv_ch=4*self.GK, out_ch=3).to(self.device)

        if ckpName is not None:
            ckp = self.load_ckp(ckpName)
            self.RelativePos_Encoder.load_state_dict(ckp['Relative_encoder'])
            self.Pred_Encoder.load_state_dict(ckp['Pred_encoder'])
            self.Pred_Decoder.load_state_dict(ckp['Pre_decoder'])
            self.skinningKernelRadiuse.data = ckp['skinRadius'].data

            if self.ifNeuralTexture:
                self.featTensor.data = ckp['featTensor'].data

        self.skinningKernelRadiuse = self.skinningKernelRadiuse.to(self.device)

        if self.ifNeuralTexture:
            self.featTensor = self.featTensor.to(self.device)

    def setNetwork_train(self):
        self.RelativePos_Encoder.train()
        self.Pred_Encoder.train()
        self.Pred_Decoder.train()
        self.skinningKernelRadiuse.requires_grad = True

        if self.ifNeuralTexture:
            self.featTensor.requires_grad = True

    def setNetwork_eval(self):
        self.RelativePos_Encoder.eval()
        self.Pred_Encoder.eval()
        self.Pred_Decoder.eval()
        self.skinningKernelRadiuse.requires_grad = False

        if self.ifNeuralTexture:
            self.featTensor.requires_grad = False

    def createOptimzer(self):
        self.Optimizer = torch.optim.Adam(params=list(self.RelativePos_Encoder.parameters()) +
                                                 list(self.Pred_Encoder.parameters()) +
                                                 list(self.Pred_Decoder.parameters()) +
                                                 [self.skinningKernelRadiuse],
                                          lr=self.ini_lr[0], betas=(0, 0.9))

        # self.Optimizer = torch.optim.Adam([{'params': list(self.RelativePos_Encoder.parameters()) +
        #                                               list(self.Pred_Encoder.parameters()) +
        #                                               list(self.Pred_Decoder.parameters()), 'lr': self.ini_lr[0]},
        #                                    {'params': [self.skinningKernelRadiuse], 'lr': self.ini_lr[1]}],
        #                                   betas=(0, 0.1))

    def adjust_learning_rate(self, itt):
        print('adjusting lr: ', itt)
        for i, param_group in enumerate(self.Optimizer.param_groups):
            param_group['lr'] = self.ini_lr[i] * (self.adfacetor[i] ** (itt // self.adlr_freq))
            print(i, param_group['lr'])

    def adjustOptlr(self, opt, itt, inilr, adfactor, adfreq):
        print('adjusting lr: ', itt)
        for i, param_group in enumerate(opt.param_groups):
            param_group['lr'] = inilr[i] * (adfactor[i] ** (itt // adfreq))
            print(i, param_group['lr'])

    def readConanicalAxises(self, GAxisFiles, device):
        x_axis = readAxisFile(GAxisFiles['x'], device)
        y_axis = readAxisFile(GAxisFiles['y'], device)
        z_axis = readAxisFile(GAxisFiles['z'], device)

        x_axis = x_axis.unsqueeze(0).repeat(self.bsize, 1, 1)
        y_axis = y_axis.unsqueeze(0).repeat(self.bsize, 1, 1)
        z_axis = z_axis.unsqueeze(0).repeat(self.bsize, 1, 1)

        return x_axis, y_axis, z_axis

    def readMapSampleFile(self, mapName, numV, device):
        levelH, levelW, levelPixelValidX, levelPixelValidY, levelMap, _ = \
            ReadSampleMap(fileName=mapName, numV=numV, outNumLevel=1, ifColor=False)

        levelH, levelW, levelPixelValidX, levelPixelValidY, levelMap = \
            levelH[0], levelW[0], levelPixelValidX[0], levelPixelValidY[0], \
            levelMap[0].type(torch.FloatTensor).to(device)

        return levelH, levelW, levelPixelValidX, levelPixelValidY, levelMap

    def do_InputBatchLoading(self, GarmFiles, BodyRTFiles):
        G_Verts = []
        #G_RelVec = []
        BB_Rotate = []
        BB_Translate = []
        BB_JointV = []
        BB_JointN = []
        GMap_relativeP = []
        for i in range(len(GarmFiles)):
            gf = GarmFiles[i]
            rtf = BodyRTFiles[i]

            gverts, gnorm = npyLoading_vert_norm(gf, self.device)
            BR, BT, BSv, BSn = load_rtvnFile(rtf, self.device)

            # save_obj('../test/t.obj', BSv.detach().cpu().numpy(), None)
            # save_obj('../test/g.obj', gverts.detach().cpu().numpy(), None)
            # exit(1)

            gpa, gpvec = Relative_PointPosition(gverts, BSv)

            gpvec_map = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                         pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                         vertMask=self.G_levelMap, hatF=torch.transpose(gpvec, 0, 1), jInfo=None)

            GMap_relativeP.append(gpvec_map.unsqueeze(0))
            BB_Rotate.append(BR.unsqueeze(0))
            BB_Translate.append(BT.unsqueeze(0))
            BB_JointV.append(BSv.unsqueeze(0))
            BB_JointN.append(BSn.unsqueeze(0))
            G_Verts.append(gverts.unsqueeze(0))
            #G_RelVec.append(gpvec.unsqueeze(0))

        GMap_relativeP = torch.cat(GMap_relativeP, dim=0)
        G_Verts = torch.cat(G_Verts, dim=0)
        BB_Rotate = torch.cat(BB_Rotate, dim=0)
        BB_Translate = torch.cat(BB_Translate, dim=0)
        BB_JointV = torch.cat(BB_JointV, dim=0)
        BB_JointN = torch.cat(BB_JointN, dim=0)
        #G_RelVec = torch.cat(G_RelVec, dim=0)

        return G_Verts, GMap_relativeP, BB_Rotate, BB_Translate, BB_JointV, BB_JointN

    def calcBatchPntJoints(self, pnts, joints, bsize):
        bpv = []
        bpa = []
        #joints = joints.to(pnts.get_device())
        for b in range(bsize):
            ax, pv = Relative_PointPosition(pnts[b, :, :], joints[b, :, :])
            bpv.append(pv.unsqueeze(0))
            bpa.append(ax.unsqueeze(0))
        bpv = torch.cat(bpv, dim=0)
        bpa = torch.cat(bpa, dim=0)
        return bpa, bpv

    def calcBatchPntJointSDFLoss(self, pnts, j_vs, j_ns, bsize):
        bjsdf = []
        for b in range(bsize):
            ax, sx, gf = Relative_force(pnts[b, :, :], j_vs[b, :, :], j_ns[b, :, :])
            val = sx * Kernel_Dist(ax, self.jSDFSigma)
            bjsdf.append(val.unsqueeze(0))
        bjsdf = torch.cat(bjsdf, dim=0)
        loss = torch.sum(bjsdf, dim=-1).mean()
        return loss

    def calcBatchPntJointSDFDist(self, pnts, gts, j_vs, j_ns, bsize):
        loss = 0.
        for b in range(bsize):
            ax, sx, g_ = Relative_force(pnts[b, :, :], j_vs[b, :, :], j_ns[b, :, :])
            gax, gsx, g_ = Relative_force(gts[b, :, :], j_vs[b, :, :], j_ns[b, :, :])
            loss = loss + self.L1DataLoss(sx, gsx)
        loss = loss / float(bsize)
        return loss

    def SDF_FeatLoss(self, dPos, device):
        sdf = self.B_SDFModel.sdf(dPos.to(device))
        val = self.B_sdfThre - sdf
        z = torch.zeros_like(val).to(device)
        loss = torch.max(z, val)
        loss = loss.mean()
        return loss

    def conf_RecLoss(self, dPos, gPos, Conf):
        Conf = Conf**2
        loss1 = self.L1DataLoss(Conf*dPos, Conf*gPos)
        loss2 = self.L1DataLoss(Conf, torch.ones_like(Conf))
        loss = loss1 + loss2
        return loss

    def smooth_GeoLoss(self, dPos):
        lv = torch.matmul(self.G_lapMatrix, dPos)
        loss = torch.mean(torch.norm(lv, dim=-1))
        return loss

    def Laplacian_Loss(self, dPos, GPos):
        lv1 = torch.matmul(self.G_lapMatrix, dPos)
        lv2 = torch.matmul(self.G_lapMatrix, GPos)
        loss = self.L1DataLoss(lv1, lv2)
        return loss

    def iter_trainNetwork(self, GarmFiles, BodyRTfs, itt):
        if itt >= 0 and itt % self.adlr_freq == 0:
            self.adjust_learning_rate(itt)

        ifRand = True

        sdfCoeff = 0.
        if itt > 500:
            sdfCoeff = 1.

        G_Verts, GMap_relativeP, BB_R, BB_T, BB_JointV, BB_JointN = \
            self.do_InputBatchLoading(GarmFiles, BodyRTfs)

        self.Optimizer.zero_grad()
        self.setNetwork_train()

        kkRadius = self.skinningKernelRadiuse[self.BBSampleLabel]

        relative_fx = self.RelativePos_Encoder(GMap_relativeP)
        fz, b, c, h, w = self.Pred_Encoder(relative_fx)

        obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                  gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

        DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D[:, :, 0:3],
                             G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                             bsize=self.bsize, device=self.device)

        dpa, dpv = self.calcBatchPntJoints(DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
        W = calcW(dpa, kkRadius)
        obj_verts = Global_GeoDeform(DD=DD, B0=self.B0Joints, BR=BB_R, BT=BB_T, GW=W,
                                     bsize=self.bsize, numV=self.G_numVs, device=self.device)

        pur_recloss = self.L1DataLoss(obj_verts, G_Verts)

        sdf_objloss = self.SDF_FeatLoss(DD, self.device)

        smooth_objloss = self.Laplacian_Loss(obj_verts, G_Verts)

        objLoss = pur_recloss + smooth_objloss + sdfCoeff * sdf_objloss

        if ifRand:
            rz = torch.randn_like(fz).to(fz)
            rand_D = self.Pred_Decoder(midz=rz, uv=self.G_KuvPoseEn,
                                       gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

            rand_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=rand_D[:, :, 0:3],
                                      G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                      bsize=self.bsize, device=self.device)

            rdpa, rdpv = \
                self.calcBatchPntJoints(rand_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
            rW = calcW(rdpa, kkRadius)
            rand_verts = Global_GeoDeform(DD=rand_DD, B0=self.B0Joints, BR=BB_R, BT=BB_T, GW=rW,
                                          bsize=self.bsize, numV=self.G_numVs, device=self.device)

            edge_rloss = self.L1DataLoss(compute_edgeLength(rand_DD, self.G_edgesID), self.G_edgeLen)

            randLoss = 0.1 * edge_rloss

        else:

            randLoss = 0.
            rand_verts = None

        gaussLoss = regulationZ(fz, 1.)
        #radiusLoss = regularizeKernalR(self.skinningKernelRadiuse)

        Loss = objLoss + randLoss + 0.001 * gaussLoss

        Loss.backward()
        self.Optimizer.step()

        return obj_verts, rand_verts, Loss, pur_recloss, gaussLoss

    def iter_evalNetwork(self, GarmFiles, BodyRTfs):
        G_Verts,  GMap_relativeP, BB_R, BB_T, BB_JointV, BB_JointN = \
            self.do_InputBatchLoading(GarmFiles, BodyRTfs)

        self.setNetwork_eval()
        with torch.no_grad():
            kkRadius = self.skinningKernelRadiuse[self.BBSampleLabel]
            relative_fx = self.RelativePos_Encoder(GMap_relativeP)
            fz, b, c, h, w = self.Pred_Encoder(relative_fx)

            obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                      gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

            DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D[:, :, 0:3],
                                 G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                 bsize=self.bsize, device=self.device)

            dpa, dpv = self.calcBatchPntJoints(DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
            W = calcW(dpa, kkRadius)
            obj_verts = Global_GeoDeform(DD=DD, B0=self.B0Joints, BR=BB_R, BT=BB_T, GW=W,
                                         bsize=self.bsize, numV=self.G_numVs, device=self.device)

            vert_objloss = self.L1DataLoss(obj_verts, G_Verts)

            return obj_verts, vert_objloss

    def run_Network(self, GarmFiles, BodyRTfs):
        G_Verts, GMap_relativeP, BB_R, BB_T, BB_JointV, BB_JointN = \
            self.do_InputBatchLoading(GarmFiles, BodyRTfs)

        self.setNetwork_eval()
        with torch.no_grad():
            kkRadius = self.skinningKernelRadiuse[self.BBSampleLabel]
            relative_fx = self.RelativePos_Encoder(GMap_relativeP)
            fz, b, c, h, w = self.Pred_Encoder(relative_fx)

            obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                      gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

            DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D[:, :, 0:3],
                                 G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                 bsize=self.bsize, device=self.device)

            dpa, dpv = self.calcBatchPntJoints(DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
            W = calcW(dpa, kkRadius)
            obj_verts = Global_GeoDeform(DD=DD, B0=self.B0Joints, BR=BB_R, BT=BB_T, GW=W,
                                         bsize=self.bsize, numV=self.G_numVs, device=self.device)

            Loss = self.L1DataLoss(obj_verts, G_Verts)

            return DD, obj_verts, fz, Loss

    def save_ckp(self, savePref, itt):
        if self.ifNeuralTexture:
            torch.save({'itter': itt, 'Relative_encoder': self.RelativePos_Encoder.state_dict(),
                        'Pred_encoder': self.Pred_Encoder.state_dict(), 'Pre_decoder': self.Pred_Decoder.state_dict(),
                        'skinRadius': self.skinningKernelRadiuse, 'featTensor': self.featTensor},
                       savePref + '_Ngen.ckp')
        else:
            torch.save({'itter': itt, 'Relative_encoder': self.RelativePos_Encoder.state_dict(),
                        'Pred_encoder': self.Pred_Encoder.state_dict(), 'Pre_decoder': self.Pred_Decoder.state_dict(),
                        'skinRadius': self.skinningKernelRadiuse},
                        savePref + '_vae64.ckp')

    def load_ckp(self, fileName):
        ckp = torch.load(fileName, map_location=lambda storage, loc: storage)
        return ckp










