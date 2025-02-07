#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))

##### TRAINING #####

# LipShiFT on cifar10 with emma loss
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
--master_port $MASTER_PORT train_lipshift.py --launcher=pytorch \
--config='configs/lipshift/cifar10.yaml'


# LipShiFT on cifar100 with emma loss
# OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
# --master_port $MASTER_PORT train_lipshift.py --launcher=pytorch \
# --config='configs/lipshift/cifar100.yaml'

# LipShiFT on Tiny ImageNet with emma loss
# OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
# --master_port $MASTER_PORT train_lipshift.py --launcher=pytorch \
# --config='configs/lipshift/tinyimagenet.yaml'



##### EVALUATION #####

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
#     --master_port $MASTER_PORT test_script.py --launcher=pytorch \
#     --config='configs/lipshift/cifar10.yaml'


