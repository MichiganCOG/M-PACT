#!/usr/bin/env bash

# C3D
wget -O models/weights/c3d_Sports1M.npy https://umich.box.com/s/agsehscj383kcyd19vudtzasdrk8zpk3
wget -O models/weights/c3d_Sports1M_finetune_UCF101.npy https://umich.box.com/s/enb6b5nc8k6wjo09jb0c4ngelxmilu74 
wget -O models/weights/sport1m_train16_128_mean.npy https://umich.box.com/s/2jz0ns0lehr41cayn5xkm0azkxtqsc44

# I3D
wget -O models/weights/i3d_rgb_kinetics.npy https://umich.box.com/s/xl06t9sb2c0qnnbh00v6dqr0fq98au0m

# ResNet50 + LSTM
wget -O models/weights/resnet50_rgb_imagenet.npy https://umich.box.com/s/ipju8t8x4hhppu3hn9y784azk3i1kxxf

# TSN (Temporal Segment Networks) 
wget -O models/weights/tsn_BNInception_ImageNet_pretrained.npy https://umich.box.com/s/0nr7pd64of7pkbml5zzexbv2zgkjw10j
wget -O models/weights/tsn_pretrained_UCF101_reordered.npy https://umich.box.com/s/xqbddtujtl79f1nqsx0apadik0inczp1
wget -O models/weights/tsn_pretrained_HMDB51_reordered.npy https://umich.box.com/s/yehf1jg8x1r7d61k6e1ee2s9bmij82fe
