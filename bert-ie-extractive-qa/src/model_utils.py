"""
模型工具模块
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorWithPadding,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """模型配置"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        fp16: bool = False,
        seed: int = 42,
        output_dir: str = "./outputs/models"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.seed = seed
        self.output_dir = output_dir
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()
    
    def save(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: str) -> 'ModelConfig':
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        set_seed(config.seed)
        logger.info(f"ModelManager initialized with {config.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None) -> Tuple:
        """加载模型和tokenizer"""
        if model_path is None:
            model_path = self.config.model_name
        
        logger.info(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.to(self.device)
        
        logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model, self.tokenizer
    
    def create_trainer(self, train_dataset, eval_dataset=None, compute_metrics=None):
        """创建Trainer"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # 计算warmup_steps
        num_training_steps = len(train_dataset) // self.config.batch_size * self.config.num_epochs
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=False,
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.config.max_length,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            # ✅ 删除 tokenizer=self.tokenizer
        )
        
        return self.trainer
    
    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        
        if self.model:
            self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        self.config.save(os.path.join(save_path, "model_config.json"))
        logger.info(f"Model saved to {save_path}")