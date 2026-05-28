from config import Config
import torch
import argparse
from process import EntityProcessor, RelationProcessor
from data_load import NERBIODataLoader
from model import BIOBertNer, BIOBertCRFNer
from train import Trainer
from evaluate import EntityEvaluator
import os

if __name__ == "__main__":
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="example", help="数据集名称")
    parser.add_argument("--model_name", type=str, default="BIOBertNer", help="模型名称")
    parser.add_argument("--save_path", type=str, default='model_en.pth', help="模型保存路径")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="训练设备")
    parser.add_argument(
        "--epochs", type=int, default=10, help="训练轮数"
    )  # 建议给个默认值
    parser.add_argument("--batch_size", type=int, default=2, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    args = parser.parse_args()

    # 2. 加载配置
    config = Config(args)
    print(config)

    # 3. 数据处理
    # 如果数据已经处理过，其实可以注释掉这行以节省时间，或者在 Processor 内部做判断
    processor = EntityProcessor(config)
    processor.process()

    # 4. 获取数据加载器
    data_loader = NERBIODataLoader(config)
    dataloaders = data_loader.get_dataloader()

    # 5. 构建模型
    if args.model_name == "BIOBertNer":
        model = BIOBertNer(config)
    elif args.model_name == "BIOBertCRFNer":
        model = BIOBertCRFNer(config)
    print(model)

    # 6. 训练模型
    trainer = Trainer(
        model, 
        updates_total=len(dataloaders["train"]) * config.epochs, 
        config=config
    )
    trainer.train(dataloaders)

    # 7. 评估模型
    evaluator = EntityEvaluator(model, config)
    evaluator.run_evaluate(dataloaders)
