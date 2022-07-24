from DynamicPred_architecture import *
import torch
from sklearn import neighbors


class Dynamic_rollout(Dynamics_PredArchi):
    def __init__(self, GAxisFiles, G0FName, G_ccName, GUVName, GK, GkimgH, GkimgW, GMapName, GVertPSampleName,
                 BSdfckp, BSampName, B0FName, bsize, device, staticCKP, dynamicCKP):
        super().__init__(GAxisFiles, G0FName, G_ccName, GUVName, GK, GkimgH, GkimgW, GMapName, GVertPSampleName,
                         BSdfckp, BSampName, B0FName, bsize, device)

        # self.RelativePos_Encoder = RelativePosEncoder(in_ch=self.BB_numSamples * 3, out_ch=128).to(self.device)
        # self.Pred_Decoder = Pred_decoder(in_ch=1024, uv_ch=4 * self.GK, out_ch=3, midSize=64).to(self.device)
        # self.skinningKernelRadiuse = torch.tensor(0.01)
        #
        # ckp = super().load_ckp(staticCKP)
        # self.skinningKernelRadiuse.data = ckp['skinRadius'].data
        # self.RelativePos_Encoder.load_state_dict(ckp['Relative_encoder'])
        # self.Pred_Decoder.load_state_dict(ckp['Pre_decoder'])
        #
        # self.DynamicForceEncoder = \
        #     DynamicDeltaEncoder(in_ch=self.BB_numSamples * 3, v_ch=6, out_ch=128).to(self.device)
        # self.DynamicEncoder = DisEncoder(in_ch=128 * 2, out_ch=1024, midSize=64).to(self.device)
        #
        # ckp = super().load_ckp(dynamicCKP)
        # self.DynamicForceEncoder.load_state_dict(ckp['Dynamic_ForceEncoder'])
        # self.DynamicEncoder.load_state_dict(ckp['Dynamic_Encoder'])

        super().createDynamicNetwork(staticCKP, dynamicCKP)

        self.RelativePos_Encoder.eval()
        self.Pred_Decoder.eval()
        self.DynamicForceEncoder.eval()
        self.DynamicEncoder.eval()
        self.skinningKernelRadiuse.requires_grad = False

        self.Rho_update = self.skinningKernelRadiuse[self.BBSampleLabel].to(self.device)
        #self.Rho_update = self.skinningKernelRadiuse.repeat(self.BB_numSamples).to(self.device)
        print(self.Rho_update.size())
        self.jSDFSigma = 0.01

        self.scale = 2
        self.ResGeoFeat = torch.zeros(3, self.G_levelH*2 // self.scale, self.G_levelW*2 // self.scale).to(self.device)
        self.ResGSampling = torch.nn.UpsamplingBilinear2d(scale_factor=self.scale).to(self.device)
        self.ResGeo_Opt = torch.optim.Adam(params=[self.ResGeoFeat], lr=1.e-3, betas=(0, 0.9))

        iniLr = 1.e-2
        self.Rho_Opt = torch.optim.Adam(params=[self.Rho_update], lr=iniLr, betas=(0, 0.9))

    def SampleResMap(self, gMIndex, gMEffi, resMap=None):
        if resMap == None:
            GeoResMap = self.ResGeoFeat.unsqueeze(0)
        else:
            GeoResMap = resMap.unsqueeze(0)
        GeoResMap = self.ResGSampling(GeoResMap).to(self.device).squeeze(0)
        GeoResMap = GeoResMap.permute(1, 2, 0)
        r = SampleMap(GeoResMap, gMIndex, gMEffi).unsqueeze(0)
        return r

    def BodyTOGarmenNearest(self, garmverts, body_verts):
        bodyTree = neighbors.KDTree(body_verts.cpu().numpy())
        dist, bInd = bodyTree.query(garmverts.cpu().numpy(), k=1)
        neiList = [i[0] for i in bInd]

        return neiList

    def calcNearestSDF(self, garmentverts, body_verts, body_normals, neiList):
        gpx = garmentverts - body_verts[neiList, :]
        bnx = body_normals[neiList, :]

        gpx = gpx.unsqueeze(-1)
        bnx = bnx.unsqueeze(1)
        sx = torch.matmul(bnx, gpx).squeeze(-1)
        sx = torch.relu(-sx)
        Loss = sx.mean()
        return Loss

    def oneRoll_genD(self, pre_gpos, pre_gvelo, pre_gacc, pre_bSeed, cur_bSeed, curr_bSNorm):
        preGMap_relative, preGMap_Force, preGMap_Dynamic = \
            super().prepareOneInputData(pre_gpos, pre_gvelo, pre_gacc, pre_bSeed, cur_bSeed, curr_bSNorm)

        with torch.no_grad():
            pre_Dynamic = self.DynamicForceEncoder(preGMap_Force, preGMap_Dynamic)
            pre_geo = self.RelativePos_Encoder(preGMap_relative)
            fx = torch.cat([pre_geo, pre_Dynamic], dim=1)

            fz, b, c, h, w = self.DynamicEncoder(fx)

            obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                      gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

            obj_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D,
                                     G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                     bsize=self.bsize, device=self.device)

            return obj_DD

    def optimizeResMap_noPropR(self, obj_DD, bVerts, bNorms, curB_rotate, curB_translate, sdfThr, maxItter):
        dpa, dpv = self.calcBatchPntJoints(obj_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
        W = calcW(dpa, self.Rho_update)
        obj_verts = Global_GeoDeform(DD=obj_DD, B0=self.B0Joints,
                                     BR=curB_rotate.unsqueeze(0), BT=curB_translate.unsqueeze(0), GW=W,
                                     bsize=self.bsize, numV=self.G_numVs, device=self.device)

        neiList = self.BodyTOGarmenNearest(obj_verts.squeeze(0), bVerts)
        sdfLoss = self.calcNearestSDF(obj_verts.squeeze(0), bVerts, bNorms, neiList)

        if sdfLoss.item() < sdfThr:
            return obj_verts, obj_DD, sdfLoss.item()

        _ResGeoFeat = torch.zeros(3, self.G_levelH * 2 // self.scale, self.G_levelW * 2 // self.scale).to(self.device)
        _ResGeo_Opt = torch.optim.Adam(params=[_ResGeoFeat], lr=1.e-3, betas=(0, 0.9))
        _ResGeoFeat.requires_grad = True

        runitt = 0
        for itt in range(maxItter + 1):
            # print(itt)
            _ResGeo_Opt.zero_grad()

            R = self.SampleResMap(gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, resMap=_ResGeoFeat)

            DD = Local_GeoDeform(G0=obj_DD.squeeze(0), GD=R,
                                 G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                 bsize=self.bsize, device=self.device)

            dpa, dpv = self.calcBatchPntJoints(DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
            W = calcW(dpa, self.Rho_update)

            obj_new = Global_GeoDeform(DD=DD, B0=self.B0Joints,
                                       BR=curB_rotate.unsqueeze(0), BT=curB_translate.unsqueeze(0), GW=W,
                                       bsize=self.bsize, numV=self.G_numVs, device=self.device)

            # sdfLoss = super().calcBatchPntJointSDFLoss(pnts=obj_new,
            #                                            j_vs=cur_bSeed.unsqueeze(0), j_ns=curr_bSNorm.unsqueeze(0),
            #                                            bsize=self.bsize)

            sdfLoss = self.calcNearestSDF(obj_new.squeeze(0), bVerts, bNorms, neiList)

            # lossSDF_C = super().SDF_FeatLoss(DD, self.device)
            LossG = self.L1DataLoss(obj_new, obj_verts)
            # LossE = self.L1DataLoss(compute_edgeLength(obj_new, self.G_edgesID),
            #                         compute_edgeLength(obj_verts, self.G_edgesID))
            smooth_objloss = self.Laplacian_Loss(obj_verts, obj_new)

            Loss = 10. * LossG + 10000. * smooth_objloss + 10000 * (sdfLoss)
            Loss.backward()
            _ResGeo_Opt.step()

            runitt = itt
            if sdfLoss.item() < sdfThr:
                break

        R = self.SampleResMap(gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, resMap=_ResGeoFeat)

        DD = Local_GeoDeform(G0=obj_DD.squeeze(0), GD=R,
                             G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                             bsize=self.bsize, device=self.device)

        dpa, dpv = self.calcBatchPntJoints(DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
        W = calcW(dpa, self.Rho_update)

        obj_new = Global_GeoDeform(DD=DD, B0=self.B0Joints,
                                   BR=curB_rotate.unsqueeze(0), BT=curB_translate.unsqueeze(0), GW=W,
                                   bsize=self.bsize, numV=self.G_numVs, device=self.device)

        sdfLoss = self.calcNearestSDF(obj_new.squeeze(0), bVerts, bNorms, neiList)

        print(runitt, ': ', sdfLoss.item())
        return obj_new, DD, sdfLoss.item()

    def optimizeResMap_1(self, obj_DD, bVerts, bNorms, curB_rotate, curB_translate, sdfThr, maxItter):
        dpa, dpv = self.calcBatchPntJoints(obj_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
        W = calcW(dpa, self.Rho_update)
        obj_verts = Global_GeoDeform(DD=obj_DD, B0=self.B0Joints,
                                     BR=curB_rotate.unsqueeze(0), BT=curB_translate.unsqueeze(0), GW=W,
                                     bsize=self.bsize, numV=self.G_numVs, device=self.device)

        # sdfLoss = super().calcBatchPntJointSDFLoss(pnts=obj_verts,
        #                                            j_vs=cur_bSeed.unsqueeze(0), j_ns=curr_bSNorm.unsqueeze(0),
        #                                            bsize=self.bsize)

        neiList = self.BodyTOGarmenNearest(obj_verts.squeeze(0), bVerts)
        sdfLoss = self.calcNearestSDF(obj_verts.squeeze(0), bVerts, bNorms, neiList)

        if sdfLoss.item() < sdfThr:
            return obj_verts, obj_DD, sdfLoss.item()

        self.ResGeoFeat.requires_grad = True
        runitt = 0
        for itt in range(maxItter+1):
            #print(itt)
            self.ResGeo_Opt.zero_grad()

            R = self.SampleResMap(gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff)

            DD = Local_GeoDeform(G0=obj_DD.squeeze(0), GD=R,
                                 G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                 bsize=self.bsize, device=self.device)

            dpa, dpv = self.calcBatchPntJoints(DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
            W = calcW(dpa, self.Rho_update)

            obj_new = Global_GeoDeform(DD=DD, B0=self.B0Joints,
                                       BR=curB_rotate.unsqueeze(0), BT=curB_translate.unsqueeze(0), GW=W,
                                       bsize=self.bsize, numV=self.G_numVs, device=self.device)

            # sdfLoss = super().calcBatchPntJointSDFLoss(pnts=obj_new,
            #                                            j_vs=cur_bSeed.unsqueeze(0), j_ns=curr_bSNorm.unsqueeze(0),
            #                                            bsize=self.bsize)

            sdfLoss = self.calcNearestSDF(obj_new.squeeze(0), bVerts, bNorms, neiList)

            #lossSDF_C = super().SDF_FeatLoss(DD, self.device)
            LossG = self.L1DataLoss(obj_new, obj_verts)
            # LossE = self.L1DataLoss(compute_edgeLength(obj_new, self.G_edgesID),
            #                         compute_edgeLength(obj_verts, self.G_edgesID))
            smooth_objloss = self.Laplacian_Loss(obj_verts, obj_new)

            Loss = 10. * LossG + 10000. * smooth_objloss + 10000 * (sdfLoss)
            Loss.backward()
            self.ResGeo_Opt.step()

            runitt = itt
            if sdfLoss.item() < sdfThr:
                break

        R = self.SampleResMap(gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff)

        DD = Local_GeoDeform(G0=obj_DD.squeeze(0), GD=R,
                             G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                             bsize=self.bsize, device=self.device)

        dpa, dpv = self.calcBatchPntJoints(DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
        W = calcW(dpa, self.Rho_update)

        obj_new = Global_GeoDeform(DD=DD, B0=self.B0Joints,
                                   BR=curB_rotate.unsqueeze(0), BT=curB_translate.unsqueeze(0), GW=W,
                                   bsize=self.bsize, numV=self.G_numVs, device=self.device)

        sdfLoss = self.calcNearestSDF(obj_new.squeeze(0), bVerts, bNorms, neiList)

        print(runitt, ': ', sdfLoss.item())
        return obj_new, DD, sdfLoss.item()

    def OneRoll_run_1(self, pre_gpos, pre_gvelo, pre_gacc, pre_bSeed,
                      curBVerts, curBNorms,
                      cur_bSeed, curr_bSNorm, curB_rotate, curB_translate, sdfThr, maxItter, ifpropagate=True):
        self.Rho_update.requires_grad = False
        self.ResGeoFeat.requires_grad = False
        obj_DD = self.oneRoll_genD(pre_gpos, pre_gvelo, pre_gacc, pre_bSeed, cur_bSeed, curr_bSNorm)
        # obj_verts, sdfVal = self.optimizeRho(obj_DD, cur_bSeed, curr_bSNorm, curB_rotate, curB_translate,
        #                                      sdfThr, maxItter)
        if ifpropagate:
            obj_verts, DD, sdfVal = self.optimizeResMap_1(obj_DD, curBVerts, curBNorms, curB_rotate, curB_translate,
                                                          sdfThr, maxItter)
        else:
            obj_verts, DD, sdfVal = self.optimizeResMap_noPropR(obj_DD, curBVerts, curBNorms, curB_rotate, curB_translate,
                                                                sdfThr, maxItter)
        return obj_verts, DD, sdfVal

    def resetIniFrameGarment(self, gpos, bSeed, new_bVerts, new_bNorms, new_bSRotate, new_bSTranslate, sdfThr, maxItter):
        self.Rho_update.requires_grad = False
        self.ResGeoFeat.requires_grad = False
        prerel_a, prerel_v = Relative_PointPosition(gpos, bSeed)

        with torch.no_grad():
            preGMap_relative = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                                pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                                vertMask=self.G_levelMap, hatF=torch.transpose(prerel_v, 0, 1), jInfo=None)
            preGMap_relative = preGMap_relative.unsqueeze(0)

            pre_geo = self.RelativePos_Encoder(preGMap_relative)
            fx = torch.cat([pre_geo, torch.zeros_like(pre_geo).to(self.device)], dim=1)
            fz, b, c, h, w = self.DynamicEncoder(fx)

            obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                      gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

            obj_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D,
                                     G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                     bsize=self.bsize, device=self.device)

        obj_verts, DD, sdfVal = self.optimizeResMap_1(obj_DD, new_bVerts, new_bNorms, new_bSRotate, new_bSTranslate,
                                                      sdfThr, maxItter)

        return obj_verts, sdfVal

