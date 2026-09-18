"""
NLP信息抽取项目
从英文文本中提取特定属性的目标词汇
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import config
from .data_utils import QADataset, DataProcessor
from .model_utils import ModelManager, ModelConfig

__all__ = [
    "config",
    "QADataset",
    "DataProcessor",
    "ModelManager",
    "ModelConfig"
]