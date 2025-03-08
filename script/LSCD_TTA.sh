#!/bin/bash

python main.py \
  --gpu=0 \
  --load-path=pretrained_models/deepall/resnet50_a2n \
  --save-path=tta/resnet50_a2n/lscd \
  --do-train=False \
  --dataset=a2n \
  \
  --TTAug \
  --TTA-bs=3 \
  --shuffled=True \
  --eval=tta_ft \
  \
  --TTA-head='em' \
  --online \
  --loss-names='lscd'\
  \
  --model=DomainAdaptor \
  --backbone=resnet50 \
  --batch-size=32 \
  --exp-num=-2 \
  --start-time=0 \
  --times=5 \
  --workers=0 \
