GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/SATS_ViTDet/base_15.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005 MODEL.BACKBONE.NAME 'build_mixt_backbone' 

# CUDA_VISIBLE_DEVICES=${GPU} python tools/train_net.py --num-gpus 1 --config-file ./configs/PascalVOC-Detection/iOD/base_15.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.005
    
# sleep 10
# CUDA_VISIBLE_DEVICES=${GPU} python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/15_p_5.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005
# # # 15 + 5 _ ft
# # # sleep 10
# CUDA_VISIBLE_DEVICES=${GPU} python tools/train_net.py --num-gpus 4 --config-file ./configs/PascalVOC-Detection/iOD/ft_15_p_5.yaml SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.005