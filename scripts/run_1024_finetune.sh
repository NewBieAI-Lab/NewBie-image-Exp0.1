#!/bin/bash
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=SYS
export OMP_NUM_THREADS=12
train_data_path='./configs/data.yaml'

model=NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP
check_path=checkpoints
batch_size=196
snr_type=lognorm
lrn=8e-5
lr=8e-5
precision=bf16
size=1024

exp_name=${model}_bs${batch_size}_lr2e-4_${precision}_NOSEQ_Newbie_RE_OPT
mkdir -p results/"$exp_name"

NNODES=1
NPROC_PER_NODE=8
MASTER_PORT=12345
NODE_RANK=0

torchrun --nproc_per_node=8 \
         --master_port=18182 \
         --master_addr=localhost \
    finetune.py \
    --master_port 18182 \
    --global_bsz_${size} 208 \
    --micro_bsz_${size} 24\
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 \
    --data_path ${train_data_path} \
    --results_dir results/"$exp_name" \
    --data_parallel sdp \
    --max_steps 871239 \
    --ckpt_every 1000 --log_every 10 \
    --precision ${precision} --grad_precision bf16 --qk_norm \
    --global_seed 20250625 \
    --num_workers 8 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    --warmup_steps 1 \
    --wd 0.05 \
    2>&1 | tee -a results/"$exp_name"/output.log
