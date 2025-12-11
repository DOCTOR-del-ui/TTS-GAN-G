#!/bin/bash

# 遍历 idx = 1 2 3 4
for idx in 1 2 3 4
do
    # 遍历 window = 48 96
    for window in 48 96
    do
        echo "==============================="
        echo "开始训练：idx=${idx}, window=${window}"
        echo "==============================="

        python train_GAN.py \
            -gen_bs 16 \
            -dis_bs 16 \
            --cmapss_idx ${idx} \
            --cmapss_window ${window} \
            --dist-url tcp://localhost:4321 \
            --dist-backend nccl \
            --world-size 1 \
            --rank 0 \
            --dataset Cmapss \
            --bottom_width 8 \
            --max_iter 20000 \
            --img_size 32 \
            --gen_model my_gen \
            --dis_model my_dis \
            --df_dim 384 \
            --d_heads 4 \
            --d_depth 3 \
            --g_depth 5,4,2 \
            --dropout 0 \
            --latent_dim 100 \
            --gf_dim 1024 \
            --num_workers 2 \
            --g_lr 0.0001 \
            --d_lr 0.0003 \
            --optimizer adam \
            --loss lsgan \
            --wd 1e-3 \
            --beta1 0.9 \
            --beta2 0.999 \
            --phi 1 \
            --batch_size 128 \
            --num_eval_imgs 5 \
            --init_type xavier_uniform \
            --n_critic 1 \
            --val_freq 20 \
            --print_freq 50 \
            --grow_steps 0 0 \
            --fade_in 0 \
            --patch_size 2 \
            --ema_kimg 500 \
            --ema_warmup 0.1 \
            --ema 0.9999 \
            --diff_aug translation,cutout,color \
            --class_name Cmaps${idx}_${window} \
            --exp_name Cmapss${idx}_${window}

        echo "完成：idx=${idx}, window=${window}"
        echo
    done
done

echo "全部训练结束！"
