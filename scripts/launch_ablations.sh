#!/usr/bin/env bash

# Example: how to launch ablation experiments using launcher_multi_archive.sh
# Update --basedir to your desired output directory

# launch an ablation on different neg_ratio values
# for ratio in 0.1 0.2 0.4 0.5; do
#     ./launcher_multi_archive.sh python ddp_train_neus.py --config configs/synthetic/mic.txt --neg_ratio $ratio --expname ablation_neg_ratio_${ratio} --basedir ./logs/ablations/neg_ratio/mic_${ratio}
# done

# launch an ablation on different seed_offset values
# for seed in 0 2 9; do
#     ./launcher_multi_archive.sh python ddp_train_neus.py --config configs/synthetic/mic.txt --seed_offset $seed --expname ablation_seed_offset_${seed} --basedir ./logs/ablations/seed_offset/mic_so_${seed}
# done

# launch all synthetic scenes
for config in chair drums hotdog lego mic; do
    $(dirname "$0")/launcher_multi_archive.sh python ddp_train_neus.py --config configs/synthetic/${config}.txt --basedir ./logs/synthetic/${config}
done