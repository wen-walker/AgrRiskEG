import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RelationDataset(Dataset):
    def __init__(self, config, tokenizer, type_path="train"):
        """
        :param config: 配置类
        :param tokenizer: BERT Tokenizer
        :param type_path: 'train' 或 'test'，用于区分读取哪个文件（如果你的数据分了的话）
                          或者直接传入具体的 json 路径
        """
        self.config = config
        self.tokenizer = tokenizer

        # 假设我们只用之前生成的 _re.json
        # 实际项目中你可能需要把 _re.json 切分成 train_re.json 和 dev_re.json
        file_path = config.PROCESSED_DATA_DIR / config.dataset / f"{type_path}_re.json"

        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        # 1. 定义插入点
        # 我们需要插入 4 个标记
        markers = [
            {"pos": item["e1_start"], "token": "[E1]"},
            {"pos": item["e1_end"], "token": "[/E1]"},
            {"pos": item["e2_start"], "token": "[E2]"},
            {"pos": item["e2_end"], "token": "[/E2]"},
        ]

        # 2. 核心逻辑：按位置倒序排列
        # reverse=True 保证我们从句子末尾开始插，不影响前面的索引
        markers.sort(key=lambda x: x["pos"], reverse=True)

        # 3. 插入标记
        text_list = list(text)  # 转成列表方便操作
        for m in markers:
            text_list.insert(m["pos"], m["token"])

        marked_text = "".join(text_list)
        # 此时 marked_text 变成了 "长江流域[E1]高温[/E1]导致..."

        # 4. Tokenization
        # 注意：tokenizer 必须已经认识 [E1] 等标记，否则它们会被拆成 '[', 'E', '1', ']'
        encoding = self.tokenizer(
            marked_text,
            padding=False,
            max_length=self.config.max_len,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding[
                "input_ids"
            ].squeeze(),  # 去掉 batch 维度 (1, Seq) -> (Seq)
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
            "has_trig": torch.tensor(item.get("has_trig", False), dtype=torch.bool),
        }
