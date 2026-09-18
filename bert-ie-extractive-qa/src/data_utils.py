"""
数据处理工具
"""

import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """数据样本"""
    id: str
    text: str
    question: str
    answers: List[str] = None
    category_zh: str = None
    category_en: str = None
    attribute_zh: str = None
    question_type: str = None
    question_id: str = None
    question_id: str = None


class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    @staticmethod
    def extract_tasks(data: Dict, is_training: bool = True) -> List[DataSample]:
        """从数据中提取任务（统一解析 task1..taskN 前缀键，训练/测试一致）"""
        samples = []
        text = data.get("text", "")
        sample_id = data.get("id", "")

        tasks = []
        for key in data:
            if key.startswith("task") and isinstance(data[key], dict):
                tasks.append(data[key])
        if not tasks and "tasks" in data:
            tasks = data["tasks"]
        elif not tasks and "task" in data:
            tasks = [data["task"]]

        for task in tasks:
            sample = DataSample(
                id=sample_id,
                text=text,
                question=task.get("question", ""),
                answers=task.get("answer", []) if is_training else None,
                category_zh=task.get("category_zh", ""),
                category_en=task.get("category_en", ""),
                attribute_zh=task.get("attribute_zh", ""),
                question_type=task.get("question_type", ""),
                question_id=task.get("question_id", "")
            )
            samples.append(sample)

        return samples


class QADataset(Dataset):
    """问答数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        is_training: bool = True,
        processor: Optional[DataProcessor] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        if processor is None:
            processor = DataProcessor()
        self.processor = processor
        
        raw_data = self.processor.load_jsonl(data_path)
        
        self.samples = []
        for data in raw_data:
            samples = self.processor.extract_tasks(data, is_training)
            self.samples.extend(samples)
        
        self.features = []
        for sample in self.samples:
            feature = self._create_feature(sample)
            if feature:
                self.features.append(feature)
        
        logger.info(f"Created {len(self.features)} features from {len(self.samples)} samples")
    
    def _create_feature(self, sample: DataSample) -> Optional[Dict]:
        """创建特征"""
        encoding = self.tokenizer(
            sample.question,
            sample.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
            return_tensors=None
        )
        
        feature = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "offset_mapping": encoding["offset_mapping"],
            "sample_id": sample.id
        }
        
        if self.is_training and sample.answers:
            start_positions, end_positions = self._find_answer_positions(
                sample.text,
                sample.answers,
                encoding["input_ids"],
                encoding["offset_mapping"]
            )
            
            if start_positions is not None and end_positions is not None:
                feature["start_positions"] = start_positions
                feature["end_positions"] = end_positions
            else:
                return None
        
        return feature
    
    def _find_answer_positions(
        self,
        text: str,
        answers: List[str],
        input_ids: List[int],
        offset_mapping: List[Tuple[int, int]]
    ) -> Tuple[Optional[int], Optional[int]]:
        """查找答案在token中的位置"""
        if not answers:
            return None, None
        
        for answer in answers:
            start_char = text.lower().find(answer.lower())
            if start_char == -1:
                continue
            
            end_char = start_char + len(answer)
            
            start_token = None
            end_token = None
            
            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == token_end == 0:
                    continue
                
                if token_start >= start_char and start_token is None:
                    start_token = idx
                
                if token_end <= end_char:
                    end_token = idx
            
            if start_token is not None and end_token is not None and start_token <= end_token:
                return start_token, end_token
        
        return None, None
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feature = self.features[idx]
        
        item = {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
        }
        
        if self.is_training and "start_positions" in feature:
            item["start_positions"] = torch.tensor(feature["start_positions"], dtype=torch.long)
            item["end_positions"] = torch.tensor(feature["end_positions"], dtype=torch.long)
        
        return item


# ✅ 自定义 collate 函数
def qa_data_collator(features):
    """
    自定义 collate 函数，正确处理 QA 数据
    """
    if not features:
        return {}
    
    # 检查是否有标签
    has_labels = "start_positions" in features[0]
    
    batch = {}
    
    # 处理 input_ids
    if "input_ids" in features[0]:
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    
    # 处理 attention_mask
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    
    # 如果有标签，处理 start_positions 和 end_positions
    if has_labels:
        batch["start_positions"] = torch.tensor([f["start_positions"] for f in features], dtype=torch.long)
        batch["end_positions"] = torch.tensor([f["end_positions"] for f in features], dtype=torch.long)
    
    return batch