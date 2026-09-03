#!/usr/bin/env bash

# Setting 1: Market1501 -> CUHK-SYSU -> DukeMTMC -> MSMT17 -> CUHK03
CUDA_VISIBLE_DEVICES=0 python continual_train.py \
  --logs-dir logs-res-setting1/ \
  --setting 1 \
  --use-fisher \
  --use-fisher-dff \
  --lambda-fim 1e-5 \
  --fisher-gamma 1.0 \
  --fisher-estimator per_sample \
  --fisher-norm per_parameter_mean \
  --fisher-max-batches 100 \
  --fisher-dff-beta 1.0
