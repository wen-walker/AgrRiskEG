import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class DataLoaderFactory:
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

        # 1. 加载 Tokenizer
        print(f"[*] Loading Tokenizer: {self.config.bert_name}...")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.bert_name, cache_dir=self.config.CACHE_DIR
        )
            # 2. 扩充词表 (添加双向虚拟锚点)
            special_tokens = ["[EXP]", "[IMP]"]
            num_added_toks = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens}
            )
            print(
                f"[*] 成功添加了 {num_added_toks} 个特殊 Token。当前词表大小: {len(self.tokenizer)}"
            )

        self.exp_id = self.tokenizer.convert_tokens_to_ids("[EXP]")
        self.imp_id = self.tokenizer.convert_tokens_to_ids("[IMP]")

    def get_dataloader(self, split, batch_size=None, shuffle=None):
        """
        获取指定数据集划分的 DataLoader
        :param split: "train", "val" 或 "test"
        :param batch_size: 覆盖配置中的 batch_size
        :param shuffle: 覆盖默认的 shuffle 策略 (默认: train为True, 其余为False)
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        if shuffle is None:
            shuffle = split == "train"

        file_path = self.config.PROCESSED_DATA_DIR / f"{split}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"⚠️ 数据文件不存在: {file_path}")

        # 实例化内部 Dataset 类
        dataset = self._CausalSetDataset(
            jsonl_file=file_path,
            tokenizer=self.tokenizer,
            exp_id=self.exp_id,
            imp_id=self.imp_id,
            max_length=self.config.max_length,
        )

        # 闭包捕获 pad_token_id
        pad_token_id = self.tokenizer.pad_token_id

        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            attention_mask = [item["attention_mask"] for item in batch]
            aux_labels = [item["aux_labels"] for item in batch]

            input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_token_id
            )
            attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            )
            aux_labels_stacked = torch.stack(aux_labels)

            targets = [item["target_dict"] for item in batch]

            return {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "aux_labels": aux_labels_stacked,
                "targets": targets,
            }

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,  # 可根据需要在 Config中添加 num_workers 配置
        )

        print(
            f"✅ 成功创建 {split} DataLoader: {len(dataset)} 个样本, batch_size={batch_size}, shuffle={shuffle}"
        )
        return dataloader

    # ==========================================
    # 内部类：Dataset
    # ==========================================
    class _CausalSetDataset(Dataset):
        def __init__(self, jsonl_file, tokenizer, exp_id, imp_id, max_length=256):
            self.data = []
            self.tokenizer = tokenizer
            self.exp_id = exp_id
            self.imp_id = imp_id
            self.max_length = max_length

            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))

        def __len__(self):
            return len(self.data)

        def _char_to_token_span(self, char_start, char_end, offset_mapping):
            token_start, token_end = -1, -1
            for idx, (os_start, os_end) in enumerate(offset_mapping):
                if idx < 3:  # 🌟 强制跳过自己拼接的 [CLS] [EXP] [IMP] 避免 0,0 干扰
                    continue
                if os_start == os_end:  # 跳过特殊的标记 (如 SEP)
                    continue

                if token_start == -1 and os_end > char_start:
                    token_start = idx
                if os_start < char_end:
                    token_end = idx

            # 🌟 加入 token_start > token_end 异常检测
            if token_start == -1 or token_end == -1 or token_start > token_end:
                return None
            return [token_start, token_end]

        def __getitem__(self, idx):
            sample = self.data[idx]
            text = sample["text"]

            encoding = self.tokenizer(
                text,
                max_length=self.max_length - 2,  # 预留2个位置给 EXP 和 IMP
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            offsets = encoding["offset_mapping"]

            new_input_ids = [input_ids[0], self.exp_id, self.imp_id] + input_ids[1:]
            new_attention_mask = [1, 1, 1] + attention_mask[1:]
            new_offsets = [(0, 0), (0, 0), (0, 0)] + offsets[1:]

            gt_labels, gt_causes, gt_triggers, gt_effects = [], [], [], []

            for tuple_data in sample["causal_tuples"]:
                is_implicit = tuple_data["type"] == "Implicit"
                label_id = 1 if is_implicit else 0

                c_span = self._char_to_token_span(
                    tuple_data["cause"][0], tuple_data["cause"][1], new_offsets
                )
                e_span = self._char_to_token_span(
                    tuple_data["effect"][0], tuple_data["effect"][1], new_offsets
                )

                if is_implicit or tuple_data["trigger"] is None:
                    t_span = [-1, -1]
                else:
                    t_span = self._char_to_token_span(
                        tuple_data["trigger"][0], tuple_data["trigger"][1], new_offsets
                    )

                if (
                    c_span is None
                    or e_span is None
                    or (not is_implicit and t_span is None)
                ):
                    continue

                gt_labels.append(label_id)
                gt_causes.append(c_span)
                gt_triggers.append(t_span)
                gt_effects.append(e_span)

            return {
                "input_ids": torch.tensor(new_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(new_attention_mask, dtype=torch.long),
                "aux_labels": torch.tensor(
                    [sample["has_exp"], sample["has_imp"]], dtype=torch.float
                ),
                "target_dict": {
                    "labels": torch.tensor(gt_labels, dtype=torch.long),
                    "cause_spans": torch.tensor(gt_causes, dtype=torch.long).reshape(
                        -1, 2
                    ),
                    "trigger_spans": torch.tensor(
                        gt_triggers, dtype=torch.long
                    ).reshape(-1, 2),
                    "effect_spans": torch.tensor(gt_effects, dtype=torch.long).reshape(
                        -1, 2
                    ),
                },
            }
