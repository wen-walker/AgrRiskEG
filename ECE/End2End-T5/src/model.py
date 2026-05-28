import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from config import Config


class T5EEModel(nn.Module):
    def __init__(self, config):
        """
        初始化 T5 模型
        :param config: 配置对象，包含 model_name 等参数
        """
        super(T5EEModel, self).__init__()
        self.config = config

        # 加载预训练的生成式模型 (Mengzi-T5)
        # T5ForConditionalGeneration 包含了 Encoder 和 Decoder
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.t5_name, cache_dir=config.CACHE_DIR
        )

        # 如果需要在模型内部进行解码（比如把生成的 ID 转回文本），可以持有 tokenizer
        # 但通常我们在外部 Evaluation 时处理解码，这里保留以备不时之需
        self.tokenizer = T5Tokenizer.from_pretrained(
            config.t5_name, cache_dir=config.CACHE_DIR
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播 - 用于训练过程
        :param input_ids: 输入文本的编码 [batch_size, seq_len]
        :param attention_mask: 输入的注意力掩码 [batch_size, seq_len]
        :param labels: 目标文本的编码 (Ground Truth) [batch_size, target_len]
        :return: (loss, logits)
        """
        # T5ForConditionalGeneration 内部会自动计算 CrossEntropyLoss
        # 前提是传入了 labels
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # output.loss 是标量 Loss
        # output.logits 是预测的分布 [batch_size, seq_len, vocab_size]
        return output.loss, output.logits

    def predict(self, input_ids, attention_mask):
        """
        推理/生成 - 用于验证或测试过程
        该方法不计算梯度，利用 Beam Search 或 Greedy Search 生成文本序列
        """
        # generate 方法会自动进行自回归解码 (Autoregressive Decoding)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.max_target_len,
            num_beams=4,  # 使用 Beam Search 提升生成质量
            length_penalty=1.0,  # 长度惩罚系数
            early_stopping=True,  # 遇到结束符提前停止
        )

        return generated_ids

    def save_pretrained(self, save_directory):
        """
        自定义保存方法，保存 HuggingFace 格式的模型和 tokenizer
        """
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
