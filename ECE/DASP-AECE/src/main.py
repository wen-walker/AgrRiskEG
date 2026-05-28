from config import Config
import torch
from transformers import AutoTokenizer
import argparse
from process import Processor
from dataload import DataLoaderFactory
from model import CausalSetExtractor
from train import Trainer
from evaluate import Evaluator

if __name__ == "__main__":
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=None, help="模型保存路径")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="训练设备")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率")
    parser.add_argument("--bert_name", type=str, default=None, help="预训练模型名称")
    args = parser.parse_args()
    # print(args)

    # 2. 加载配置
    config = Config(args)
    # print(config)

    # 3. 数据处理
    processor = Processor(config)
    processor.process()

    # 4. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.bert_name, cache_dir=config.CACHE_DIR
    )
    special_tokens = ["[EXP]", "[IMP]"] # 双向虚拟锚点
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    print(f"[*] 成功添加了 {num_added} 个特殊 Token。当前词表大小: {len(tokenizer)}")
    # 在初始化所有组件前，一次性将派生配置写入 config，之后不再改变
    config.tokenizer_len = len(tokenizer)

    # 5. 获取数据加载器
    factory = DataLoaderFactory(config, tokenizer=tokenizer)
    train_loader = factory.get_dataloader("train")
    val_loader = factory.get_dataloader("val")
    test_loader = factory.get_dataloader("test")

    # 6. 构建模型
    model = CausalSetExtractor(config)
    print(model)

    # 7. 训练模型
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    trainer.train()

    # 8. 评估模型
    model_save_path = config.MODEL_SAVE_DIR / "causal_set_extractor.pth"
    evaluator = Evaluator(
        config=config,
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        checkpoint_path=model_save_path
    )
    evaluator.run_evaluate()
