"""
项目主入口
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config


def run_command(cmd: str):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="NLP QA Project Main Entry")
    parser.add_argument("mode", choices=["train", "evaluate", "predict", "create_val", "info"], 
                        help="Mode to run")
    parser.add_argument("--train_file", help="Training data path")
    parser.add_argument("--eval_file", help="Evaluation data path")
    parser.add_argument("--test_file", help="Test data path")
    parser.add_argument("--model_path", help="Model path")
    parser.add_argument("--output_file", help="Output file path")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--model_name", help="Model name")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    
    args = parser.parse_args()
    
    if args.mode == "info":
        config.print_paths()
        return
    
    if args.mode == "create_val":
        from create_validation import main as create_val_main
        create_val_main()
        return
    
    config.setup_directories()
    
    if args.mode == "train":
        cmd = f"python src/train.py"
        cmd += f" --train_file {args.train_file or config.TRAIN_FILE}"
        cmd += f" --eval_file {args.eval_file or config.VALIDATION_FILE}"
        cmd += f" --model_name {args.model_name or config.MODEL_NAME}"
        cmd += f" --batch_size {args.batch_size or config.BATCH_SIZE}"
        cmd += f" --num_epochs {args.num_epochs or config.NUM_EPOCHS}"
        run_command(cmd)
    
    elif args.mode == "evaluate":
        model_path = args.model_path or os.path.join(config.MODEL_DIR, "final_model")
        eval_file = args.eval_file or config.VALIDATION_FILE
        
        if not os.path.exists(eval_file):
            print(f"❌ 验证集不存在: {eval_file}")
            print("请先运行: python main.py create_val")
            return
        
        cmd = f"python src/evaluate.py --model_path {model_path} --data_file {eval_file}"
        cmd += f" --output_file {args.output_file or os.path.join(config.PRED_DIR, 'evaluation_results.json')}"
        cmd += f" --batch_size {args.batch_size or config.BATCH_SIZE}"
        run_command(cmd)
    
    elif args.mode == "predict":
        model_path = args.model_path or os.path.join(config.MODEL_DIR, "final_model")
        test_file = args.test_file or config.TEST_A_FILE
        
        if not os.path.exists(test_file):
            print(f"❌ 测试文件不存在: {test_file}")
            print("请确保 data/test_a/test_a.jsonl 文件存在")
            return
        
        cmd = f"python src/predict.py --model_path {model_path} --test_file {test_file}"
        cmd += f" --output_file {args.output_file or os.path.join(config.PRED_DIR, 'predictions.jsonl')}"
        cmd += f" --batch_size {args.batch_size or config.BATCH_SIZE}"
        run_command(cmd)


if __name__ == "__main__":
    main()