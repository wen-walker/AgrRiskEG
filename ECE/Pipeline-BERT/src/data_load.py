from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from functools import partial
from tqdm import tqdm

class NERBIODataLoader:
    def __init__(self, config):
        self.config = config
    # 动态 padding
    def collate_fn(self, batch, tokenizer):
        # 1. 提取句子字符列表和标签
        texts_chs = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        # 2. 使用分词器进行编码，动态 padding
        tokenized_inputs = tokenizer(
            texts_chs,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.config.max_len,  # 添加最大长度限制
            return_tensors="pt",
        )
        # 3. 对齐标签
        # Tokenizer 之后，输入变成了 [CLS]...[SEP][PAD]...，我们需要构造对应长度的 labels
        new_labels = []
        for i, ori_label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = []
            for word_id in word_ids:
                if word_id is None:
                    # word_id 为 None 的时候，表示 [CLS]、[SEP]、[PAD] 等特殊字符
                    aligned_labels.append(-100)
                else:
                    aligned_labels.append(ori_label[word_id])
            new_labels.append(aligned_labels)
        # 4. 将对齐后的标签转为张量，并放入input字典中
        tokenized_inputs["labels"] = torch.tensor(new_labels)
        return tokenized_inputs

    def data_loader(self, dataset):
        # 1. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(self.config.bert_name, cache_dir=str(self.config.CACHE_DIR))
        # 2. 使用 partial 固定 collate_fn 中的 tokenizer 参数
        # 这样 DataLoader 调用 collate_fn(batch) 时，会自动带上 tokenizer
        collate_func = partial(self.collate_fn, tokenizer=tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=collate_func
        )
        return dataloader

    def get_dataloader(self):
        dataset_dict = load_dataset('json', data_files={
            'train': str(self.config.PROCESSED_DATA_DIR / '{}/train_processed.json'.format(self.config.dataset)),
            'val': str(self.config.PROCESSED_DATA_DIR / '{}/val_processed.json'.format(self.config.dataset)),
            'test': str(self.config.PROCESSED_DATA_DIR / '{}/test_processed.json'.format(self.config.dataset))
        })
        print(dataset_dict)
        train_dataset = dataset_dict['train']
        val_dataset = dataset_dict['val']
        test_dataset = dataset_dict['test']
        # 使用 tqdm 包装数据加载过程
        dataloaders = {}
        for dataset, name in tqdm([(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')], 
                                desc="🚀 Creating dataloaders", 
                                colour='cyan'):
            dataloaders[name] = self.data_loader(dataset)
        # 获取第一个批次进行测试
        # first_batch = next(iter(dataloaders['val']))
        # print(first_batch)
        return dataloaders


import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from re_dataset import RelationDataset


class RelationCollateFn:
    """
    处理动态 Padding 的辅助类
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # 1. 提取 batch 中的各个字段
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        token_type_ids = [item["token_type_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        has_trig = [item["has_trig"] for item in batch]

        # 2. 动态 Padding (batch_first=True 返回 [Batch, Seq])
        # input_ids 用 pad_token_id 补齐
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        # attention_mask 用 0 补齐
        attention_mask_padded = pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        # token_type_ids 用 0 补齐
        token_type_ids_padded = pad_sequence(
            token_type_ids, batch_first=True, padding_value=0
        )

        # labels 堆叠
        labels_stacked = torch.stack(labels)
        # has_trig 堆叠
        has_trig_stacked = torch.stack(has_trig)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "token_type_ids": token_type_ids_padded,
            "labels": labels_stacked,
            "has_trig": has_trig_stacked,
        }


class RelationDataLoader:
    """
    封装后的 DataLoader 类
    自动初始化 Dataset，并配置动态 Padding
    """

    def __init__(self, config, tokenizer):
        """
        :param config: 配置类实例
        """
        self.config = config
        self.tokenizer = tokenizer
    def data_loader(self, type_path):
        """
        创建指定类型的数据加载器
        :param type_path: 'train' / 'val' / 'test'
        :return: DataLoader instance
        """
        # 1. 初始化 Dataset
        dataset_instance = RelationDataset(self.config, self.tokenizer, type_path=type_path)

        # 2. 获取 tokenizer（从实例变量获取）
        tokenizer = self.tokenizer
        # 3. 确定 shuffle 策略
        shuffle = type_path == "train"

        # 4. 初始化 Collate Function
        collate_fn = RelationCollateFn(tokenizer)

        # 5. 创建 DataLoader
        dataloader = DataLoader(
            dataset=dataset_instance,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),  # 如果有 GPU，开启 pin_memory 加速
        )

        return dataloader

    def get_dataloader(self):
        """
        返回包含 train、val 和 test 的数据加载器字典
        """
        dataloaders = {}
        for type_path in tqdm(["train", "val", "test"], desc="🚀 Creating dataloaders", colour='cyan'):
            # 检查文件是否存在
            import os

            file_path = (
                self.config.PROCESSED_DATA_DIR
                / self.config.dataset
                / f"{type_path}_re.json"
            )
            if os.path.exists(file_path):
                dataloaders[type_path] = self.data_loader(type_path)

        return dataloaders
