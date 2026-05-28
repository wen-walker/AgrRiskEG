import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq


class EET5DataLoader(object):
    def __init__(self, config):
        self.config = config
        # 加载 Tokenizer (Mengzi-T5)
        self.tokenizer = T5Tokenizer.from_pretrained(
            config.t5_name, cache_dir=config.CACHE_DIR
        )

        # 定义 Data Collator
        # 它的作用是动态 Padding，并将 label 中的 pad token id 设置为 -100
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=(
                8 if config.fp16 else None
            ),  # 如果开启混合精度训练，设置为8倍数有助于加速
        )

    def preprocess_function(self, examples):
        """
        将文本数据转换为模型输入的 ID
        """
        inputs = examples["input_text"]
        targets = examples["target_text"]

        # 1. 处理输入 (Source)
        model_inputs = self.tokenizer(
            inputs, max_length=self.config.max_source_len, truncation=True
        )

        # 2. 处理输出 (Target)
        # 关键点：使用 text_target 参数来告诉 tokenizer 这是输出序列
        labels = self.tokenizer(
            text_target=targets, max_length=self.config.max_target_len, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_dataloader(self):
        """
        构建并返回 train, val, test 的 DataLoader
        """
        # 构建文件路径字典
        data_files = {}
        processed_dir = self.config.PROCESSED_DATA_DIR / self.config.dataset

        # 检查并添加存在的文件
        if (processed_dir / "train_processed.json").exists():
            data_files["train"] = str(processed_dir / "train_processed.json")
        if (processed_dir / "val_processed.json").exists():
            data_files["validation"] = str(processed_dir / "val_processed.json")
        if (processed_dir / "test_processed.json").exists():
            data_files["test"] = str(processed_dir / "test_processed.json")

        # 1. 使用 HuggingFace Datasets 加载 JSON
        # print(f"Loading datasets from: {data_files}")
        raw_datasets = load_dataset("json", data_files=data_files)

        # 2. 批量进行 Tokenization
        tokenized_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["input_text", "target_text"],  # 处理完后移除原始文本列
            desc="Running tokenizer on dataset",
        )

        # 3. 设置格式为 PyTorch Tensor
        tokenized_datasets.set_format(type="torch")

        dataloaders = {}

        # 4. 构建 DataLoader
        if "train" in tokenized_datasets:
            dataloaders["train"] = DataLoader(
                tokenized_datasets["train"],
                shuffle=True,
                collate_fn=self.data_collator,
                batch_size=self.config.batch_size,
            )

        if "validation" in tokenized_datasets:
            dataloaders["val"] = DataLoader(
                tokenized_datasets["validation"],
                shuffle=False,  # 验证集不需要 shuffle
                collate_fn=self.data_collator,
                batch_size=self.config.batch_size,
            )

        if "test" in tokenized_datasets:
            dataloaders["test"] = DataLoader(
                tokenized_datasets["test"],
                shuffle=False,
                collate_fn=self.data_collator,
                batch_size=self.config.batch_size,
            )

        return dataloaders
