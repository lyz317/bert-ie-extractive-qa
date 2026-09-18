"""
训练脚本
"""

import os
import sys
import logging
import argparse

from transformers import set_seed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data_utils import QADataset, qa_data_collator
from src.model_utils import ModelManager, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    return {"exact_match": 0.0, "f1": 0.0}


def train(args):
    set_seed(config.SEED)
    
    model_config = ModelConfig(
        model_name=args.model_name or config.MODEL_NAME,
        max_length=config.MAX_LENGTH,
        batch_size=args.batch_size or config.BATCH_SIZE,
        learning_rate=args.learning_rate or config.LEARNING_RATE,
        num_epochs=args.num_epochs or config.NUM_EPOCHS,
        output_dir=config.MODEL_DIR
    )
    
    manager = ModelManager(model_config)
    model, tokenizer = manager.load_model()
    
    logger.info("Loading training data...")
    train_dataset = QADataset(
        args.train_file or config.TRAIN_FILE,
        tokenizer,
        max_length=config.MAX_LENGTH,
        is_training=True
    )
    
    eval_dataset = None
    if args.eval_file and os.path.exists(args.eval_file):
        logger.info("Loading evaluation data...")
        eval_dataset = QADataset(
            args.eval_file,
            tokenizer,
            max_length=config.MAX_LENGTH,
            is_training=True
        )
    
    # ✅ 直接使用自定义 collate
    trainer = manager.create_trainer(
        train_dataset,
        eval_dataset,
        compute_metrics=compute_metrics if eval_dataset else None,
        data_collator=qa_data_collator  # ✅ 添加这行
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    final_model_path = os.path.join(config.MODEL_DIR, "final_model")
    manager.save_model(final_model_path)
    
    if eval_dataset:
        logger.info("Final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train QA model")
    parser.add_argument("--train_file", type=str, help="Training data file path")
    parser.add_argument("--eval_file", type=str, help="Evaluation data file path")
    parser.add_argument("--model_name", type=str, help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()