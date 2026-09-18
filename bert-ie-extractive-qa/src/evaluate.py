"""
评估脚本
"""

import json
import logging
import argparse
import os
import sys
import re

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DataCollatorWithPadding

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data_utils import QADataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_f1(pred: str, true: str) -> float:
    pred_tokens = normalize_text(pred).split()
    true_tokens = normalize_text(true).split()
    
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(true_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    logger.info(f"Loading evaluation data from {args.data_file}")
    dataset = QADataset(
        args.data_file,
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
    
    ground_truths = {}
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sample_id = data.get("id", "")
                for key in data:
                    if key.startswith("task") and isinstance(data[key], dict):
                        task = data[key]
                        gid = (sample_id, task.get("question_id", ""))
                        answers = task.get("answer", [])
                        if answers:
                            ground_truths[gid] = answers
    
    total_em = 0.0
    total_f1 = 0.0
    total_samples = 0
    global_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
                pred = tokenizer.convert_tokens_to_string(tokens).strip()
                
                sample = dataset.samples[global_idx]
                global_idx += 1
                gid = (sample.id, sample.question_id)
                if gid in ground_truths and ground_truths[gid]:
                    best_f1 = 0.0
                    best_em = 0.0
                    for gt in ground_truths[gid]:
                        f1 = compute_f1(pred, gt)
                        em = 1.0 if normalize_text(pred) == normalize_text(gt) else 0.0
                        best_f1 = max(best_f1, f1)
                        best_em = max(best_em, em)
                    total_f1 += best_f1
                    total_em += best_em
                    total_samples += 1
    
    avg_em = total_em / total_samples if total_samples > 0 else 0.0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
    
    results = {
        "exact_match": avg_em,
        "f1": avg_f1,
        "total_samples": total_samples
    }
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Total Samples: {total_samples}")
    logger.info(f"  Exact Match: {avg_em:.4f}")
    logger.info(f"  F1 Score: {avg_f1:.4f}")
    
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--output_file", type=str, help="Path to save results")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == "__main__":
    main()