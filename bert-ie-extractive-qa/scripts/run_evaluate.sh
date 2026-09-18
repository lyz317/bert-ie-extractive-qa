#!/bin/bash

# 评估脚本
python main.py evaluate \
    --model_path ./outputs/models/final_model \
    --eval_file ./data/validation.jsonl \
    --output_file ./outputs/evaluation_results.json