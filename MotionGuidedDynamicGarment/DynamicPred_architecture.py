from StaticPred_architecture import Statics_PredArchi
from DataIO import *
from Displace_model import *
from geometry import *


class Dynamics_PredArchi(Statics_PredArchi):
    def __init__(self, GAxisFiles, G0FName, G_ccName, GUVName, GK, GkimgH, GkimgW, GMapName, GVertPSampleName,
                 BSdfckp, BSampName, B0FName, bsize, device):
        super().__init__(GAxisFiles, G0FName, G_ccName, GUVName, GK, GkimgH, GkimgW, GMapName, GVertPSampleName,
                         B0FName, BSdfckp, BSampName, bsize, device)
        self.SecondOrder = True

        self.BG_kernelSigma = 0.04

        self.DynamicForceEncoder = None
        self.DynamicEncoder = None

        self.DynamicOptimizer = None

    def loadStatic_PretrainingNetwork(self, ckpName):
        self.RelativePos_Encoder = RelativePosEncoder(in_ch=self.BB_numSamples * 3, out_ch=128).to(self.device)
        self.Pred_Decoder = Pred_decoder(in_ch=1024, uv_ch=4 * self.GK, out_ch=3, midSize=64).to(self.device)
        self.skinningKernelRadiuse = torch.ones(self.NumSeedLables) * 0.01

        ckp = super().load_ckp(ckpName)
        self.skinningKernelRadiuse.data = ckp['skinRadius'].data
        self.RelativePos_Encoder.load_state_dict(ckp['Relative_encoder'])
        self.Pred_Decoder.load_state_dict(ckp['Pre_decoder'])

        self.skinningKernelRadiuse = self.skinningKernelRadiuse.to(self.device)
        self.skinningKernelRadiuse.requires_grad = False
        self.RelativePos_Encoder.requires_grad = False
        self.Pred_Decoder.requires_grad = False

    def createDynamicNetwork(self, static_ckp, dyn_ckp=None):
        self.loadStatic_PretrainingNetwork(static_ckp)
        if self.SecondOrder:
            v_ch = 6
        else:
            v_ch = 3

        self.DynamicForceEncoder = \
            DynamicDeltaEncoder(in_ch=self.BB_numSamples * 3, v_ch=v_ch, out_ch=128).to(self.device)
        self.DynamicEncoder = DisEncoder(in_ch=128*2, out_ch=1024, midSize=64).to(self.device)

        if dyn_ckp is not None:
            ckp = super().load_ckp(dyn_ckp)
            self.DynamicForceEncoder.load_state_dict(ckp['Dynamic_ForceEncoder'])
            self.DynamicEncoder.load_state_dict(ckp['Dynamic_Encoder'])

    def createDynamicOptimizer(self):
        self.DynamicOptimizer = torch.optim.Adam(params=list(self.DynamicForceEncoder.parameters()) +
                                                        list(self.DynamicEncoder.parameters()),
                                                 lr=1.e-4, betas=(0, 0.9), amsgrad=True)

    def set_Dynamic_train(self):
        self.DynamicForceEncoder.train()
        self.DynamicEncoder.train()

    def set_Dynamic_eval(self):
        self.DynamicForceEncoder.eval()
        self.DynamicEncoder.eval()

    def BatchLoading_inputPrepare(self, preGarmFiles, preBodySeedFiles, BodySeedFiles, numF):
        preGarm_verts = []
        preGMap_relative = []
        preGMap_Force = []
        preGMap_Dynamic = []
        preBB_Rotate = []
        preBB_Translate = []
        BB_Rotate = []
        BB_Translate = []

        for i in range(numF):
            pre_gposit, pre_gnorms, pre_gveloc, pre_gaccle = npyLoading_pos_norm_vel_acc(preGarmFiles[i], self.device)

            preGarm_verts.append(pre_gposit.detach().clone().unsqueeze(0))

            pre_BR, pre_BT, pre_BSverts, pre_BSnorms = load_rtvnFile(preBodySeedFiles[i], self.device)
            prerel_a, prerel_v = Relative_PointPosition(pre_gposit, pre_BSverts)
            relvMap = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                       pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                       vertMask=self.G_levelMap, hatF=torch.transpose(prerel_v, 0, 1), jInfo=None)

            BR, BT, BSverts, BSnorms = load_rtvnFile(BodySeedFiles[i], self.device)
            BSnorms = normalize_vetors(BSnorms)
            gpa, gsx, gforce = Relative_force(pre_gposit, BSverts, BSnorms)
            gforce = sigmaForce(gforce, prerel_a, self.BG_kernelSigma)

            gpf_map = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                       pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                       vertMask=self.G_levelMap, hatF=torch.transpose(gforce, 0, 1), jInfo=None)

            if self.SecondOrder:
                dynVec = torch.cat([pre_gveloc, pre_gaccle], dim=-1)
            else:
                dynVec = pre_gveloc
            gdyn_map = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                        pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                        vertMask=self.G_levelMap, hatF=torch.transpose(dynVec, 0, 1), jInfo=None)

            preGMap_relative.append(relvMap.unsqueeze(0))
            preGMap_Force.append(gpf_map.unsqueeze(0))
            preGMap_Dynamic.append(gdyn_map.unsqueeze(0))

            preBB_Rotate.append(pre_BR.unsqueeze(0))
            preBB_Translate.append(pre_BT.unsqueeze(0))
            BB_Rotate.append(BR.unsqueeze(0))
            BB_Translate.append(BT.unsqueeze(0))

        preGarm_verts = torch.cat(preGarm_verts, dim=0)
        preGMap_relative = torch.cat(preGMap_relative, dim=0)
        preGMap_Force = torch.cat(preGMap_Force, dim=0)
        preGMap_Dynamic = torch.cat(preGMap_Dynamic, dim=0)

        preBB_Rotate = torch.cat(preBB_Rotate, dim=0)
        preBB_Translate = torch.cat(preBB_Translate, dim=0)
        BB_Rotate = torch.cat(BB_Rotate, dim=0)
        BB_Translate = torch.cat(BB_Translate, dim=0)

        return preGarm_verts, preGMap_relative, preGMap_Force, preGMap_Dynamic, \
               preBB_Rotate, preBB_Translate, BB_Rotate, BB_Translate

    def BatchLoading_targetFrameGarment(self, currGarmentFiles):
        Garm_verts = []
        for gf in currGarmentFiles:
            verts, norms = npyLoading_vert_norm(gf, self.device)
            Garm_verts.append(verts.unsqueeze(0))
        Garm_verts = torch.cat(Garm_verts, dim=0)
        return Garm_verts

    def Iter_trainDynNetwork(self, preGarmFiles, preBodySeedFiles, preGZ,
                             curGarmentFiles, curBodySeedFiles, currGZ, itt):
        preGarm_verts, preGMap_relative, preGMap_Force, preGMap_Dynamic, \
        preBB_Rotate, preBB_Translate, BB_Rotate, BB_Translate = \
            self.BatchLoading_inputPrepare(preGarmFiles, preBodySeedFiles, curBodySeedFiles, self.bsize)

        G_verts = self.BatchLoading_targetFrameGarment(curGarmentFiles)

        if itt >= 0 and itt % self.adlr_freq == 0:
            super().adjustOptlr(self.DynamicOptimizer, itt,
                                inilr=self.ini_lr, adfactor=self.adfacetor, adfreq=self.adlr_freq)

        self.DynamicOptimizer.zero_grad()
        self.set_Dynamic_train()

        kkRadius = self.skinningKernelRadiuse[self.BBSampleLabel]

        pre_Dynamic = self.DynamicForceEncoder(preGMap_Force, preGMap_Dynamic)
        pre_geo = self.RelativePos_Encoder(preGMap_relative)

        '''---------------Reconstruct prev frame prediction--------------'''
        prefx = torch.cat([pre_geo, torch.zeros_like(pre_Dynamic).to(pre_Dynamic)], dim=1)
        prefz, preb, prec, preh, prew = self.DynamicEncoder(prefx)

        pre_D = self.Pred_Decoder(midz=prefz, uv=self.G_KuvPoseEn,
                                  gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff,
                                  b=preb, c=prec, h=preh, w=prew)

        preobj_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=pre_D,
                                    G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                    bsize=self.bsize, device=self.device)

        predpa, predpv = self.calcBatchPntJoints(preobj_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
        preW = calcW(predpa, kkRadius)
        preobj_verts = Global_GeoDeform(DD=preobj_DD, B0=self.B0Joints, BR=preBB_Rotate, BT=preBB_Translate, GW=preW,
                                        bsize=self.bsize, numV=self.G_numVs, device=self.device)

        prevert_objloss = self.L1DataLoss(preobj_verts, preGarm_verts)
        #presdf_objloss = self.SDF_FeatLoss(preobj_DD, self.device)

        prez_loss = self.L1DataLoss(preGZ, prefz)

        preobjLoss = prevert_objloss

        '''------------Dynamic prediction---------------'''
        fx = torch.cat([pre_geo, pre_Dynamic], dim=1)
        fz, b, c, h, w = self.DynamicEncoder(fx)

        obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                  gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

        obj_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D,
                                 G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                 bsize=self.bsize, device=self.device)

        dpa, dpv = self.calcBatchPntJoints(obj_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
        W = calcW(dpa, kkRadius)
        obj_verts = Global_GeoDeform(DD=obj_DD, B0=self.B0Joints, BR=BB_Rotate, BT=BB_Translate, GW=W,
                                     bsize=self.bsize, numV=self.G_numVs, device=self.device)

        vert_objloss = self.L1DataLoss(obj_verts, G_verts)
        #sdf_objloss = self.SDF_FeatLoss(obj_DD, self.device)

        z_loss = self.L1DataLoss(currGZ, fz)

        objLoss = vert_objloss

        '''---------------Opt loss----------------'''

        gaussLoss = 0.5 * (regulationZ(fz, 1.) + regulationZ(prefz, 1.))
        rec_Loss = 0.5 * (objLoss + preobjLoss)
        code_Loss = 0.5 * (z_loss + prez_loss)

        Loss = rec_Loss + code_Loss + 0.001 * gaussLoss

        Loss.backward()
        self.DynamicOptimizer.step()

        return preobj_verts, obj_verts, \
               Loss, vert_objloss, code_Loss, gaussLoss

    def Iter_evalDynNetwork(self, preGarmFiles, preBodySeedFiles, curGarmentFiles, curBodySeedFiles):
        preGarm_verts, preGMap_relative, preGMap_Force, preGMap_Dynamic, \
        preBB_Rotate, preBB_Translate, BB_Rotate, BB_Translate = \
            self.BatchLoading_inputPrepare(preGarmFiles, preBodySeedFiles, curBodySeedFiles, self.bsize)

        G_verts = self.BatchLoading_targetFrameGarment(curGarmentFiles)

        self.set_Dynamic_eval()
        with torch.no_grad():
            kkRadius = self.skinningKernelRadiuse[self.BBSampleLabel]
            pre_Dynamic = self.DynamicForceEncoder(preGMap_Force, preGMap_Dynamic)
            pre_geo = self.RelativePos_Encoder(preGMap_relative)

            '''---------------Reconstruct prev frame prediction--------------'''
            prefx = torch.cat([pre_geo, torch.zeros_like(pre_Dynamic).to(pre_Dynamic)], dim=1)
            prefz, preb, prec, preh, prew = self.DynamicEncoder(prefx)

            pre_D = self.Pred_Decoder(midz=prefz, uv=self.G_KuvPoseEn,
                                      gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff,
                                      b=preb, c=prec, h=preh, w=prew)

            preobj_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=pre_D,
                                        G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                        bsize=self.bsize, device=self.device)

            predpa, predpv = self.calcBatchPntJoints(preobj_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1),
                                                     self.bsize)
            preW = calcW(predpa, kkRadius)
            preobj_verts = Global_GeoDeform(DD=preobj_DD, B0=self.B0Joints, BR=preBB_Rotate, BT=preBB_Translate,
                                            GW=preW,
                                            bsize=self.bsize, numV=self.G_numVs, device=self.device)

            prevert_objloss = self.L1DataLoss(preobj_verts, preGarm_verts)

            '''------------Dynamic prediction---------------'''
            fx = torch.cat([pre_geo, pre_Dynamic], dim=1)
            fz, b, c, h, w = self.DynamicEncoder(fx)

            obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                      gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

            obj_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D,
                                     G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                     bsize=self.bsize, device=self.device)

            dpa, dpv = self.calcBatchPntJoints(obj_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
            W = calcW(dpa, kkRadius)
            obj_verts = Global_GeoDeform(DD=obj_DD, B0=self.B0Joints, BR=BB_Rotate, BT=BB_Translate, GW=W,
                                         bsize=self.bsize, numV=self.G_numVs, device=self.device)

            vert_objloss = self.L1DataLoss(obj_verts, G_verts)

            vert_objloss = 0.5 * (vert_objloss + prevert_objloss)

            return preobj_verts, obj_verts, vert_objloss

    def prepareOneInputData(self, pre_gpos, pre_gvelo, pre_gacc, pre_bSeed, cur_bSeed, curr_bSNorm):
        prerel_a, prerel_v = Relative_PointPosition(pre_gpos, pre_bSeed)
        preGMap_relative = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                            pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                            vertMask=self.G_levelMap, hatF=torch.transpose(prerel_v, 0, 1), jInfo=None)
        preGMap_relative = preGMap_relative.unsqueeze(0)

        #bsample_v, bsample_n = samplePnts(self.BBsampleIDs, self.BBsampleUVs, cur_body, curr_bodyNorm)
        curr_bSNorm = normalize_vetors(curr_bSNorm)
        gpa, gsx, gforce = Relative_force(pre_gpos, cur_bSeed, curr_bSNorm)
        gforce = sigmaForce(gforce, prerel_a, self.BG_kernelSigma)

        preGMap_Force = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                         pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                         vertMask=self.G_levelMap, hatF=torch.transpose(gforce, 0, 1), jInfo=None)
        preGMap_Force = preGMap_Force.unsqueeze(0)

        if self.SecondOrder:
            dynVec = torch.cat([pre_gvelo, pre_gacc], dim=-1)
        else:
            dynVec = pre_gvelo
        preGMap_Dynamic = self.poseMapping(H=self.G_levelH, W=self.G_levelW,
                                           pX=self.G_levelPixelValidX, pY=self.G_levelPixelValidY,
                                           vertMask=self.G_levelMap, hatF=torch.transpose(dynVec, 0, 1), jInfo=None)
        preGMap_Dynamic = preGMap_Dynamic.unsqueeze(0)

        return preGMap_relative, preGMap_Force, preGMap_Dynamic

    def dynamicRun(self, pre_gpos, pre_gvelo, pre_gacc, pre_bSeed, cur_bSeed, curr_bSNorm, curB_rotate, curB_translate):
        preGMap_relative, preGMap_Force, preGMap_Dynamic = \
            self.prepareOneInputData(pre_gpos, pre_gvelo, pre_gacc, pre_bSeed, cur_bSeed, curr_bSNorm)

        self.set_Dynamic_eval()
        with torch.no_grad():
            kkRadius = self.skinningKernelRadiuse[self.BBSampleLabel]
            pre_Dynamic = self.DynamicForceEncoder(preGMap_Force, preGMap_Dynamic)
            pre_geo = self.RelativePos_Encoder(preGMap_relative)
            fx = torch.cat([pre_geo, pre_Dynamic], dim=1)

            fz, b, c, h, w = self.DynamicEncoder(fx)

            obj_D = self.Pred_Decoder(midz=fz, uv=self.G_KuvPoseEn,
                                      gMIndex=self.G_vertPixelIndex, gMEffi=self.G_vertPixelCoeff, b=b, c=c, h=h, w=w)

            obj_DD = Local_GeoDeform(G0=self.G0_TVerts, GD=obj_D,
                                     G0TV=self.G0_TAxis, G0BV=self.G0_BAxis, G0NV=self.G0_NAxis,
                                     bsize=self.bsize, device=self.device)

            dpa, dpv = self.calcBatchPntJoints(obj_DD, self.B0Joints.unsqueeze(0).repeat(self.bsize, 1, 1), self.bsize)
            W = calcW(dpa, kkRadius)
            obj_verts = Global_GeoDeform(DD=obj_DD, B0=self.B0Joints,
                                         BR=curB_rotate.unsqueeze(0), BT=curB_translate.unsqueeze(0), GW=W,
                                         bsize=self.bsize, numV=self.G_numVs, device=self.device)

            return obj_verts

    def save_dynckp_d(self, savePref, itt):
        torch.save({'itter': itt,
                    'Dynamic_ForceEncoder': self.DynamicForceEncoder.state_dict(),
                    'Dynamic_Encoder': self.DynamicEncoder.state_dict()}, savePref + '_Temp_dyn.ckp')
