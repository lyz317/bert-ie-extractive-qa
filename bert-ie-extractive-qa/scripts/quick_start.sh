#!/bin/bash

echo "🚀 NLP信息抽取项目 - 快速开始"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo ""
echo "1️⃣ 查看路径配置..."
python main.py info

echo ""
echo "2️⃣ 创建验证集..."
python main.py create_val

echo ""
echo "3️⃣ 开始训练..."
python main.py train

echo ""
echo "4️⃣ 生成预测结果..."
python main.py predict

echo ""
echo "✅ 完成！预测结果保存在: outputs/predictions/predictions.jsonl"