import torch
import torch.nn as nn
from transformers import AutoModel


# ==========================================
# 1. 跨度指针提取头 (Span Pointer Head)
# ==========================================
class SpanPointerHead(nn.Module):
    """
    接收 Query 特征和 序列(H) 特征，通过点乘注意力计算每个 Token 成为 Start 或 End 的概率
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.q_start_proj = nn.Linear(hidden_size, hidden_size)
        self.q_end_proj = nn.Linear(hidden_size, hidden_size)
        self.k_start_proj = nn.Linear(hidden_size, hidden_size)
        self.k_end_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, queries, memory_features):
        """
        queries: [batch_size, num_queries, hidden_size]
        memory_features: [batch_size, seq_len, hidden_size]
        返回: start_logits, end_logits -> [batch_size, num_queries, seq_len]
        """
        q_start = self.q_start_proj(queries)
        q_end = self.q_end_proj(queries)
        k_start = self.k_start_proj(memory_features)
        k_end = self.k_end_proj(memory_features)

        start_logits = torch.bmm(q_start, k_start.transpose(1, 2))
        end_logits = torch.bmm(q_end, k_end.transpose(1, 2))

        return start_logits, end_logits


# ==========================================
# 2. 核心模型：农业复杂因果联合抽取器
# ==========================================
class CausalSetExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_queries_exp = config.num_queries_exp
        self.num_queries_imp = config.num_queries_imp
        self.num_queries = self.num_queries_exp + self.num_queries_imp

        # -------------------------------------
        # Phase 1: 双向语义感知 Backbone
        # -------------------------------------
        # 必须开启 output_hidden_states 才能获取最后4层
        self.encoder = AutoModel.from_pretrained(
            config.bert_name, cache_dir=config.CACHE_DIR, output_hidden_states=True
        )

        # 兼容新加的 [EXP] 和 [IMP] (需从 config 中获取动态词表长度)
        if hasattr(config, "tokenizer_len"):
            self.encoder.resize_token_embeddings(config.tokenizer_len)
        else:
            print(
                "⚠️ 警告: config 中未找到 tokenizer_len，请确保在实例化模型前将其赋值！"
            )

        # 🌟 新增：如果使用最后4层，需要增加一个线性映射降维
        if config.use_bert_last_4_layers:
            self.bert_proj = nn.Linear(self.hidden_size * 4, self.hidden_size)

        # -------------------------------------
        # Phase 2: 并行特征解耦与辅助监督分支 (ISPD)
        # -------------------------------------
        self.aux_exp_head = nn.Linear(self.hidden_size, 1)
        self.aux_imp_head = nn.Linear(self.hidden_size, 1)

        # -------------------------------------
        # Phase 3: 动态查询路由与集合解码
        # -------------------------------------
        self.query_generator_exp = nn.Linear(
            self.hidden_size, self.hidden_size * self.num_queries_exp
        )
        self.query_generator_imp = nn.Linear(
            self.hidden_size, self.hidden_size * self.num_queries_imp
        )
        self.query_pos_embed = nn.Parameter(
            torch.randn(1, self.num_queries, self.hidden_size)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # -------------------------------------
        # Phase 3.5: 多头输出网络
        # -------------------------------------
        self.class_head = nn.Linear(self.hidden_size, 3)
        self.cause_head = SpanPointerHead(self.hidden_size)
        self.trigger_head = SpanPointerHead(self.hidden_size)
        self.effect_head = SpanPointerHead(self.hidden_size)

    def _get_encoder_hidden_states(self, outputs):
        """根据配置决定使用最后一层还是最后4层拼接"""
        if self.config.use_bert_last_4_layers:
            # hidden_states 是一个 tuple，包含 embedding 层和 12 个 transformer 层的输出
            # 取最后4层: [-1], [-2], [-3], [-4]
            last_4_layers = outputs.hidden_states[-4:]
            # 在最后一维拼接: [B, L, 768*4]
            H = torch.cat(last_4_layers, dim=-1)
            # 降维投影回 768: [B, L, 768]
            H = self.bert_proj(H)
            return H
        else:
            return outputs.last_hidden_state

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        # ==========================================
        # Phase 1: 编码与前置锚点提取
        # ==========================================
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = self._get_encoder_hidden_states(outputs)

        h_exp = H[:, 1, :]
        h_imp = H[:, 2, :]

        # ==========================================
        # Phase 2: 辅助监督分支输出
        # ==========================================
        aux_exp_logits = self.aux_exp_head(h_exp).squeeze(-1)
        aux_imp_logits = self.aux_imp_head(h_imp).squeeze(-1)

        # ==========================================
        # Phase 3: 动态查询路由
        # ==========================================
        queries_exp = self.query_generator_exp(h_exp).view(
            batch_size, self.num_queries_exp, self.hidden_size
        )
        queries_imp = self.query_generator_imp(h_imp).view(
            batch_size, self.num_queries_imp, self.hidden_size
        )

        queries_init = torch.cat([queries_exp, queries_imp], dim=1)
        queries_init = queries_init + self.query_pos_embed

        padding_mask = attention_mask == 0

        queries_updated = self.decoder(
            tgt=queries_init, memory=H, memory_key_padding_mask=padding_mask
        )

        # ==========================================
        # Phase 4: 集合预测特征头输出
        # ==========================================
        class_logits = self.class_head(queries_updated)

        cause_start, cause_end = self.cause_head(queries_updated, H)
        trigger_start, trigger_end = self.trigger_head(queries_updated, H)
        effect_start, effect_end = self.effect_head(queries_updated, H)

        return {
            "pred_logits": class_logits,
            "pred_cause_span": (cause_start, cause_end),
            "pred_trigger_span": (trigger_start, trigger_end),
            "pred_effect_span": (effect_start, effect_end),
            "h_exp": h_exp,
            "h_imp": h_imp,
            "aux_exp_logits": aux_exp_logits,
            "aux_imp_logits": aux_imp_logits,
        }

