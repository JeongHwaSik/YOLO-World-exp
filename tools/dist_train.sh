#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}


# Train Command

# YOLO-World-S (benchmark)
# ./tools/dist_train.sh configs/experiment/yolo_world_s_dual_vlpan_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.py 8 --amp --use-wandb

# YOLO-World-S w/ T2I Attention Fusion (MaxSigmoid)
# ./tools/dist_train.sh configs/experiment/yolo_world_s_vlattnfusion_l2norm_8gpus_obj365v1_train_lvis_minival.py 8 --amp --use-wandb

# YOLO-World-S remove (4),(3),(2),(1) fusion
# ./tools/dist_train.sh configs/experiment/yolo_world_s_dual_vlpan_remove4321_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.py 8 --amp --use-wandb

# YOLO-World-S remove (4),(3) fusion
# ./tools/dist_train.sh configs/experiment/yolo_world_s_dual_vlpan_remove43_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.py 8 --amp --use-wandb

# YOLO-World-S  w/ Deep Fusion
# ./tools/dist_train.sh configs/experiment/yolo_world_s_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival.py 8 --amp --use-wandb

# YOLO-World-S  w/ Multi-Scale Deep Fusion
# ./tools/dist_train.sh configs/experiment/yolo_world_s_multi_scale_deepfusion_l2norm_8gpus_obj365v1_train_lvis_minival.py 8 --amp --use-wandb



# YOLO-World-M (benchmark)
# ./tools/dist_train.sh configs/experiment/yolo_world_m_dual_vlpan_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival.py 8 --amp --use-wandb --resume work_dirs/yolo_world_m_dual_vlpan_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival/epoch_22.pth
