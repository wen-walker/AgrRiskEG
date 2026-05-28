from config import Config
import torch
import argparse
from process import Processor
from data_load import EET5DataLoader
from model import T5EEModel
from train import Trainer
from evaluate import Evaluator
from transformers import T5ForConditionalGeneration  # 用于加载权重
from pathlib import Path

if __name__ == "__main__":
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="example", help="数据集名称")
    parser.add_argument("--save_path", type=str, default=None, help="模型保存路径")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="训练设备")
    parser.add_argument(
        "--epochs", type=int, default=20, help="训练轮数"
    )
    parser.add_argument("--batch_size", type=int, default=12, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    args = parser.parse_args()

    # 2. 加载配置
    config = Config(args)
    print("🛠️ 当前配置:", config)

    # 3. 数据处理
    # 如果数据已经处理过，其实可以注释掉这行以节省时间，或者在 Processor 内部做判断
    processor = Processor(config)
    processor.process()

    # 4. 获取数据加载器
    data_loader = EET5DataLoader(config)
    dataloaders = data_loader.get_dataloader()
    train_loader = dataloaders.get("train", None)
    val_loader = dataloaders.get("val", None)
    test_loader = dataloaders.get("test", None)
    # print(next(iter(test_loader)))

    # 5. 构建模型 (初始化状态)
    model = T5EEModel(config)
    # print(model) # 模型结构通常很长，可以先注释掉

    # 6. 训练模型
    trainer = Trainer(config, model, train_loader, val_loader)
    trainer.train()

    # 7. 评估模型 (使用测试集 + 最佳权重)
    # 确定评估模型路径：优先用 --save_path，否则用 Trainer 默认的 best_model
    save_path = Path(config.save_path) if config.save_path else None
    best_model_dir = config.MODEL_SAVE_DIR / "best_model"

    if save_path and (save_path / "config.json").exists():
        # save_path 直接指向权重目录
        eval_load_path = save_path
    elif save_path and (save_path / "best_model" / "config.json").exists():
        # save_path 指向 models/ 父目录，实际权重在 best_model/ 下
        eval_load_path = save_path / "best_model"
    elif best_model_dir.exists():
        eval_load_path = best_model_dir
    else:
        eval_load_path = None

    if eval_load_path:
        print(f"📂 加载评估模型权重: {eval_load_path}")
        model.model = T5ForConditionalGeneration.from_pretrained(eval_load_path)
        model.model.to(device)
    else:
        print("⚠️ 未找到已保存的模型权重，将使用未训练的模型进行评估")

    evaluator = Evaluator(config, model, test_loader)
    evaluator.evaluate()
