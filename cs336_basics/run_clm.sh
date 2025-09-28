#!/bin/bash
cd /home/kuangph/CS336-Assignment1
export PYTHONPATH=/home/kuangph/CS336-Assignment1:$PYTHONPATH
python /home/kuangph/CS336-Assignment1/cs336_basics/run_clm.py \
    --d_model 512 \
    --num_heads 8 \
    --d_ff 2048 \
    --vocab_size 32000 \
    --num_layers 6\
    --max_seq_length 512 \
    --seq_length 256 \
    --batch_size 32 \
    --theta 100000 \
    --device cuda \
    --num_epochs 4 \
    --lr 1e-4 \
    --lr_min 1e-5 \
    --warmup_ratio 0.1 \
    --warmfix_ratio 0.9 \
    --chunk_size 500000 \
    --vocab_path /home/kuangph/CS336-Assignment1/data/vocab_32000.txt \
    --merges_path /home/kuangph/CS336-Assignment1/data/merges_32000.txt \
    --special_tokens "<|endoftext|>" \
    --corpus_path /home/kuangph/CS336-Assignment1/data/276M.txt \
    --save_path model_checkpoints/ \
    --log_interval 100 \
    --save_interval 50 \
