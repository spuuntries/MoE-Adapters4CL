#!/bin/bash

# DIL: Office-Home dataset with MoE-Adapters
# 4 domains: Art, Clipart, Product, Real World
# 65 classes shared across all domains

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/domain \
    --config-name officehome-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/officehome.yaml"
