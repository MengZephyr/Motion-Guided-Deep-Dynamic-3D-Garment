# Motion-Guided-Deep-Dynamic-3D-Garment

## Introduction

This repository contains the implemetation of [Motion-Guided-Deep-Dynamic-3D-Garment]()

If you want to setup your own training, you need to prepare the dataset composed of the sequence of the body dynamics and the corresponding garment dynamics. Then,
>> To get the required information and saved in the folder of '/uv_body/' and '/uv_garment/', please refer the code in [./uv_rasterRender](https://github.com/MengZephyr/Motion-Guided-Deep-Dynamic-3D-Garment/tree/main/uv_rasterRender) 
>> To get the sign distance field of the body shape at conanical post, please refer to [IGR: Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR). 
>> To get the sequence in '/garment_pnva/', please refer to [garment_Data_Prep3.py](https://github.com/MengZephyr/Motion-Guided-Deep-Dynamic-3D-Garment)
>> To get the sequence in '/body_RTVN/', please refer to [body_Data_Prepare.py](https://github.com/MengZephyr/Motion-Guided-Deep-Dynamic-3D-Garment/tree/main/MotionGuidedDynamicGarment)
>> To train the static generative garment deformation network, please refer to [train_static_reconstruction.py](https://github.com/MengZephyr/Motion-Guided-Deep-Dynamic-3D-Garment/tree/main/MotionGuidedDynamicGarment)
>> To get the garment correponding latent code sequence, please refer to [run_static_reconstruction.py](https://github.com/MengZephyr/Motion-Guided-Deep-Dynamic-3D-Garment/tree/main/MotionGuidedDynamicGarment)
>> To train the dynamic encoder, please refer to [train_dynamic_prediction.py](https://github.com/MengZephyr/Motion-Guided-Deep-Dynamic-3D-Garment/tree/main/MotionGuidedDynamicGarment)
>> To rollout the dynamic prediction with collision handling, please refer to [run_dynamic_prediction.py](https://github.com/MengZephyr/Motion-Guided-Deep-Dynamic-3D-Garment/tree/main/MotionGuidedDynamicGarment)
