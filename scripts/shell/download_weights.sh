#!/usr/bin/env bash

# C3D
wget -O models/weights/c3d_Sports1M.npy https://umich.box.com/shared/static/agsehscj383kcyd19vudtzasdrk8zpk3.npy 
wget -O models/weights/c3d_Sports1M_finetune_UCF101.npy https://umich.box.com/shared/static/enb6b5nc8k6wjo09jb0c4ngelxmilu74.npy 
wget -O models/weights/sport1m_train16_128_mean.npy https://umich.box.com/shared/static/2jz0ns0lehr41cayn5xkm0azkxtqsc44.npy

# I3D
wget -O models/weights/i3d_rgb_kinetics.npy https://umich.box.com/shared/static/xl06t9sb2c0qnnbh00v6dqr0fq98au0m.npy

# ResNet50 + LSTM
wget -O models/weights/resnet50_rgb_imagenet.npy https://umich.box.com/shared/static/ipju8t8x4hhppu3hn9y784azk3i1kxxf.npy

# TSN (Temporal Segment Networks) 
wget -O models/weights/tsn_BNInception_ImageNet_pretrained.npy https://umich.box.com/shared/static/0nr7pd64of7pkbml5zzexbv2zgkjw10j.npy
wget -O models/weights/tsn_pretrained_UCF101_reordered.npy https://umich.box.com/shared/static/xqbddtujtl79f1nqsx0apadik0inczp1.npy
wget -O models/weights/tsn_pretrained_HMDB51_reordered.npy https://umich.box.com/shared/static/yehf1jg8x1r7d61k6e1ee2s9bmij82fe.npy
