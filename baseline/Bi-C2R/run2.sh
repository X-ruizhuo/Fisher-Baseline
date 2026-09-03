#!/usr/bin/env bash

# Setting 2: DukeMTMC -> MSMT17 -> Market1501 -> CUHK-SYSU -> CUHK03
CUDA_VISIBLE_DEVICES=1 python continual_train.py \
  --logs-dir logs-res-setting2/ \
  --setting 2 \
  --use-fisher \
  --use-fisher-dff \
  --lambda-fim 1e-5 \
  --fisher-gamma 1.0 \
  --fisher-estimator per_sample \
  --fisher-norm per_parameter_mean \
  --fisher-max-batches 100 \
  --fisher-dff-beta 1.0
