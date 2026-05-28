from config import Config
import torch
import argparse
from process import RelationProcessor
from re_dataset import RelationDataset
from data_load import RelationDataLoader
from model import RelationModel
from train import Trainer
from evaluate import RelationEvaluator
import os
from transformers import AutoTokenizer

if __name__ == "__main__":
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="example", help="数据集名称")
    parser.add_argument("--model_name", type=str, default="BIOBertNer", help="模型名称")
    parser.add_argument("--save_path", type=str, default='model_re.pth', help="模型保存路径")
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
    # print(config)

    # 3. 数据处理
    # # 如果数据已经处理过，其实可以注释掉这行以节省时间，或者在 Processor 内部做判断
    relation_processor = RelationProcessor(config)
    relation_processor.process()

    # 4.设置分词器
    # 添加特殊标记到 Tokenizer 的词表中，重要！！后面模型需要调整 Embedding 大小
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir=str(config.CACHE_DIR))
    special_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 5. 获取数据加载器
    data_loader = RelationDataLoader(config, tokenizer)
    dataloaders = data_loader.get_dataloader()
    # print(next(iter(dataloaders['val'])))

    # 5. 构建模型
    model = RelationModel(config, tokenizer)
    # print(model)

    # 6. 训练模型
    trainer = Trainer(
        model,
        updates_total=len(dataloaders["train"]) * config.epochs,
        config=config
    )
    trainer.train(dataloaders)

    # # 7. 评估模型
    evaluator = RelationEvaluator(model, config)
    evaluator.run_evaluate(dataloaders)