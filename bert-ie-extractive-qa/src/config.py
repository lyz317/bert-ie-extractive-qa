"""
全局配置文件
根据你的实际数据结构调整
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """全局配置"""
    
    # ============ Hugging Face 镜像（解决网络问题） ============
    HF_ENDPOINT: str = "https://hf-mirror.com"
    
    # ============ 项目根目录 ============
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ============ 数据路径 ============
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    
    # 训练数据
    TRAIN_DIR: str = os.path.join(DATA_DIR, "train")
    TRAIN_FILE: str = os.path.join(TRAIN_DIR, "train.jsonl")
    
    # 测试数据A
    TEST_A_DIR: str = os.path.join(DATA_DIR, "test_a")
    TEST_A_FILE: str = os.path.join(TEST_A_DIR, "test_a.jsonl")
    SUBMIT_FILE: str = os.path.join(TEST_A_DIR, "submit.jsonl")
    
    # 验证集（从训练集划分，放在data目录下）
    VALIDATION_FILE: str = os.path.join(DATA_DIR, "validation.jsonl")
    
    # ============ 输出路径 ============
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
    MODEL_DIR: str = os.path.join(OUTPUT_DIR, "models")
    PRED_DIR: str = os.path.join(OUTPUT_DIR, "predictions")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")
    
    # ============ 模型配置 ============
    MODEL_NAME: str = "bert-base-uncased"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 8                    # 减小批次避免显存不足
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    WARMUP_RATIO: float = 0.1
    WEIGHT_DECAY: float = 0.01
    GRADIENT_ACCUMULATION_STEPS: int = 1
    FP16: bool = False                     # CPU不支持，改为False
    SEED: int = 42
    
    # ============ 训练配置 ============
    EVAL_STRATEGY: str = "epoch"
    SAVE_STRATEGY: str = "epoch"
    LOAD_BEST_MODEL_AT_END: bool = True
    EARLY_STOPPING_PATIENCE: int = 3
    LOGGING_STEPS: int = 100
    SAVE_TOTAL_LIMIT: int = 2
    
    # ============ 设备配置 ============
    DEVICE: str = "cpu"                    # 改为cpu，避免CUDA问题
    
    # ============ 验证集划分比例 ============
    VAL_SPLIT_RATIO: float = 0.2
    
    @classmethod
    def setup_directories(cls):
        """创建必要的目录"""
        for dir_path in [cls.DATA_DIR, cls.TRAIN_DIR, cls.TEST_A_DIR,
                         cls.OUTPUT_DIR, cls.MODEL_DIR, cls.PRED_DIR, cls.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def print_paths(cls):
        """打印所有路径，方便调试"""
        print("=" * 60)
        print("项目路径配置:")
        print(f"  项目根目录: {cls.PROJECT_ROOT}")
        print(f"  数据目录: {cls.DATA_DIR}")
        print(f"  训练数据: {cls.TRAIN_FILE} ({'✅ 存在' if os.path.exists(cls.TRAIN_FILE) else '❌ 不存在'})")
        print(f"  验证数据: {cls.VALIDATION_FILE} ({'✅ 存在' if os.path.exists(cls.VALIDATION_FILE) else '❌ 不存在'})")
        print(f"  测试数据A: {cls.TEST_A_FILE} ({'✅ 存在' if os.path.exists(cls.TEST_A_FILE) else '❌ 不存在'})")
        print(f"  提交样例: {cls.SUBMIT_FILE} ({'✅ 存在' if os.path.exists(cls.SUBMIT_FILE) else '❌ 不存在'})")
        print(f"  输出目录: {cls.OUTPUT_DIR}")
        print(f"  模型目录: {cls.MODEL_DIR}")
        print(f"  预测目录: {cls.PRED_DIR}")
        print("=" * 60)


# 创建配置实例
config = Config()

# 设置目录
config.setup_directories()

# 设置Hugging Face镜像
os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT