#!/bin/bash

python main.py \
  --dataset='a2n' \
  --save-path='pretrained_models/deepall/resnet50_a2n' \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  \
  --model='erm' \
  --backbone='resnet50' \
  --batch-size=32 \
  --num-epoch=30 \
  \
  --exp-num=-2 \
  --start-time=0 \
  --times=5 \
  --eval-step=1 \
  --scheduler='step' \
  --lr-decay-gamma=0.1 \
 \
  --train='deepall' \
  --eval='deepall' \
  --loader='normal' \
  --workers=0 \