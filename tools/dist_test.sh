#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}



# Test Command

# YOLO-World-S (benchmark)
# ./tools/dist_test.sh configs/experiment/yolo_world_s_dual_vlpan_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.py checkpoints/yolo_world_s_dual_vlpan_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.pth 1

# YOLO-World-S w/ T2I Attention Fusion (MaxSigmoid)
# ./tools/dist_test.sh configs/experiment/yolo_world_s_vlattnfusion_l2norm_8gpus_obj365v1_train_lvis_minival.py checkpoints/yolo_world_s_vlattnfusion_l2norm_8gpus_obj365v1_train_lvis_minival.pth 2

# YOLO-World-S remove (4),(3),(2),(1) fusion
# ./tools/dist_test.sh configs/experiment/yolo_world_s_dual_vlpan_remove4321_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.py checkpoints/epoch_1.pth 2

# YOLO-World-S remove (4),(3) fusion
# ./tools/dist_test.sh configs/experiment/yolo_world_s_dual_vlpan_remove43_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.py work_dirs/yolo_world_s_dual_vlpan_remove43_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival/epoch_8.pth 1

# YOLO-World-S  w/ Deep Fusion
# ./tools/dist_test.sh configs/experiment/yolo_world_s_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival.py checkpoints/yolo_world_s_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival.pth 1


# YOLO-World-S  w/ Multi-Scale Deep Fusion
# ./tools/dist_test.sh configs/experiment/yolo_world_s_multi_scale_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival.py work_dirs/yolo_world_s_multi_scale_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival/epoch_1.pth 1


# YOLO-World-S  w/ Deep Fusion x2
# ./tools/dist_test.sh configs/experiment/yolo_world_s_multi_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival.py work_dirs/yolo_world_s_multi_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival/epoch_1.pth 1


