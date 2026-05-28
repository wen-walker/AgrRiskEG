import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class ModelOutput(object):
    def __init__(self, logits, loss, labels):
        self.logits = logits
        self.loss = loss
        self.labels = labels

class BIOBertNer(nn.Module):
    def __init__(self, config):
        super(BIOBertNer, self).__init__()
        self.config = config
        # 1. 加载预训练 BERT 模型
        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir=str(config.CACHE_DIR))
        self.fc = nn.Linear(self.bert.config.hidden_size, len(config.labels))
        self.dropout = nn.Dropout(0.1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: (batch_size, seq_len)
        :param token_type_ids: (batch_size, seq_len)
        :return: logits: (batch_size, seq_len, num_classes)
        """
        bert_embs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.config.use_bert_last_4_layers
        )
        if self.config.use_bert_last_4_layers:
            word_reps = torch.stack(
                bert_embs.hidden_states[-4:], dim=-1
            ).mean(dim=-1)  # (batch_size, seq_len, hidden_size)
        else:
            word_reps = bert_embs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(word_reps)
        logits = self.fc(sequence_output)  # (batch_size, seq_len, num_classes)
        loss = None
        if labels is not None:
            loss = self.criterion(logits.permute(0, 2, 1), labels)
        modelOutput = ModelOutput(logits, loss, labels)
        return modelOutput

class BIOBertCRFNer(nn.Module):
    def __init__(self, config):
        super(BIOBertCRFNer, self).__init__()
        self.config = config
        # 1. 加载预训练 BERT 模型
        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir=str(config.CACHE_DIR))
        self.fc = nn.Linear(self.bert.config.hidden_size, len(config.labels))
        self.crf = CRF(len(config.labels), batch_first=True)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: (batch_size, seq_len)
        :param token_type_ids: (batch_size, seq_len)
        :return: logits: (batch_size, seq_len, num_classes)
        """
        bert_embs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.config.use_bert_last_4_layers
        )
        if self.config.use_bert_last_4_layers:
            word_reps = torch.stack(
                bert_embs.hidden_states[-4:], dim=-1
            ).mean(dim=-1)  # (batch_size, seq_len, hidden_size)
        else:
            word_reps = bert_embs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(word_reps)
        sequence_output = self.fc(sequence_output)  # (batch_size, seq_len, num_classes)
        logits = self.crf.decode(sequence_output)
        logits = torch.tensor(logits)
        loss = None
        if labels is not None:
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            loss = -self.crf(sequence_output, labels_for_crf, mask=attention_mask.bool(), reduction='mean')
        modelOutput = ModelOutput(logits, loss, labels)
        return modelOutput



class RelationModel(nn.Module):
    """
    关系抽取模型
    原理：通过强制模型“看见”被 [E1] 和 [E2] 标记的词向量，模型能更直接地学习这两个词之间的关系，而不仅仅是猜测句子的语气。
    """
    def __init__(self, config, tokenizer):
        super(RelationModel, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(
            config.bert_name, cache_dir=str(config.CACHE_DIR)
        )
        self.bert.resize_token_embeddings(len(tokenizer))

        # 记录特殊 Token 的 ID，方便后续查找位置
        self.e1_id = tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_id = tokenizer.convert_tokens_to_ids("[E2]")

        self.dropout = nn.Dropout(0.1)

        # 输入维度变大了 3 倍 (CLS + E1 + E2)
        self.fc = nn.Linear(self.bert.config.hidden_size * 3, 2)
        class_weights = torch.tensor([1.0, 2.0]).to(config.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 1. 获取序列的隐藏状态 (Batch, Seq, Hidden)
        sequence_output = outputs.last_hidden_state

        # 2. 获取 [CLS] 向量 (Batch, Hidden)
        cls_output = outputs.pooler_output

        # 3. 提取 [E1] 和 [E2] 位置的向量
        # 这是一个难点，因为每个 batch 中 [E1] 的位置不一样
        # 我们需要找到 input_ids 中等于 self.e1_id 的索引

        batch_size, seq_len = input_ids.shape

        # 找 [E1] 的位置: (Batch,)
        # nonzero() 返回索引，我们只需要列索引
        # 注意：这里假设每句话一定有且仅有一个 [E1]，数据预处理要保证这点
        e1_mask = input_ids == self.e1_id  # (Batch, Seq) boolean
        e2_mask = input_ids == self.e2_id

        # 利用 mask 提取向量
        # 逻辑：将 mask 扩展为 (Batch, Seq, Hidden) 然后相乘求和
        # 这样非 E1 位置都是 0，相加后就是 E1 的向量

        # (Batch, Seq, 1)
        e1_mask_expanded = e1_mask.unsqueeze(-1).float()
        e2_mask_expanded = e2_mask.unsqueeze(-1).float()

        # sum(dim=1) 变成 (Batch, Hidden)
        e1_feature = torch.sum(sequence_output * e1_mask_expanded, dim=1)
        e2_feature = torch.sum(sequence_output * e2_mask_expanded, dim=1)

        # 4. 拼接特征
        # (Batch, Hidden * 3)
        concat_feature = torch.cat([cls_output, e1_feature, e2_feature], dim=-1)

        output = self.dropout(concat_feature)
        logits = self.fc(output)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return ModelOutput(logits, loss, labels)
