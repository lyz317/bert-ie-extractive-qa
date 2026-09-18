"""
预测脚本
"""

import os
import json
import logging
import argparse
import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DataCollatorWithPadding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data_utils import QADataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    logger.info(f"Loading test data from {args.test_file}")
    dataset = QADataset(
        args.test_file,
        tokenizer,
        max_length=config.MAX_LENGTH,
        is_training=False
    )
    
    # ✅ DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config.MAX_LENGTH
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator
    )
    
    predictions = []
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_indices = torch.argmax(start_logits, dim=1)
            end_indices = torch.argmax(end_logits, dim=1)
            
            for i in range(input_ids.size(0)):
                start_idx = start_indices[i].item()
                end_idx = end_indices[i].item()
                
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i][start_idx:end_idx+1])
                answer = tokenizer.convert_tokens_to_string(tokens).strip()
                
                # 取真实样本 ID 与问题 ID（is_training=False 时 features 与 samples 顺序一致）
                sample = dataset.samples[global_idx]
                global_idx += 1
                predictions.append({
                    "id": sample.id,
                    "question_id": sample.question_id,
                    "answer": [answer] if answer else []
                })
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    logger.info(f"Predictions saved to {args.output_file}")
    logger.info(f"Total predictions: {len(predictions)}")


def main():
    parser = argparse.ArgumentParser(description="Make predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_file", type=str, default=None, help="Output file")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = os.path.join(config.PRED_DIR, "predictions.jsonl")
    
    predict(args)


if __name__ == "__main__":
    main()