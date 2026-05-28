import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm


# ==========================================
# 1. 损失函数定义
# ==========================================
class SetCriterion(nn.Module):
    def __init__(
        self,
        num_queries_exp,
        num_queries_imp,
        weight_cls,
        weight_span,
        weight_len,
        weight_aux,
        weight_ispd,
    ):
        super().__init__()
        self.num_queries_exp = num_queries_exp
        self.num_queries_imp = num_queries_imp
        self.num_queries = num_queries_exp + num_queries_imp

        self.weight_cls = weight_cls
        self.weight_span = weight_span
        self.weight_len = weight_len
        self.weight_aux = weight_aux
        self.weight_ispd = weight_ispd

        self.span_loss_fn = nn.CrossEntropyLoss()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def _asymmetric_match(self, outputs, targets):
        bs = outputs["pred_logits"].shape[0]
        indices = []

        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            out_c_s = outputs["pred_cause_span"][0][b].softmax(-1)
            out_c_e = outputs["pred_cause_span"][1][b].softmax(-1)
            out_e_s = outputs["pred_effect_span"][0][b].softmax(-1)
            out_e_e = outputs["pred_effect_span"][1][b].softmax(-1)

            tgt = targets[b]
            if len(tgt["labels"]) == 0:
                indices.append(([], []))
                continue

            tgt_ids = tgt["labels"]
            exp_gt_idx = (tgt_ids == 0).nonzero(as_tuple=True)[0]
            imp_gt_idx = (tgt_ids == 1).nonzero(as_tuple=True)[0]

            match_q_idx, match_gt_idx = [], []

            def match_group(q_indices, gt_indices):
                if len(gt_indices) == 0:
                    return
                gt_list = gt_indices.tolist()

                p_cls = out_prob[q_indices][:, tgt_ids[gt_list]]
                p_c_s = out_c_s[q_indices][:, tgt["cause_spans"][gt_list, 0]]
                p_c_e = out_c_e[q_indices][:, tgt["cause_spans"][gt_list, 1]]
                p_e_s = out_e_s[q_indices][:, tgt["effect_spans"][gt_list, 0]]
                p_e_e = out_e_e[q_indices][:, tgt["effect_spans"][gt_list, 1]]

                cost = -self.weight_cls * p_cls - self.weight_span * (
                    p_c_s + p_c_e + p_e_s + p_e_e
                )
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

                for r, c in zip(row_ind, col_ind):
                    match_q_idx.append(q_indices[r])
                    match_gt_idx.append(gt_list[c])

            match_group(list(range(0, self.num_queries_exp)), exp_gt_idx)
            match_group(list(range(self.num_queries_exp, self.num_queries)), imp_gt_idx)
            indices.append((match_q_idx, match_gt_idx))

        return indices

    def forward(self, outputs, targets, aux_labels):
        indices = self._asymmetric_match(outputs, targets)
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        bs = pred_logits.shape[0]

        target_classes = torch.full(
            (bs, self.num_queries), 2, dtype=torch.long, device=device
        )

        for b, (q_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) > 0:
                target_classes[b, q_idx] = targets[b]["labels"][gt_idx].to(device)

        loss_cls = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)

        # 🌟 修改：初始化为 tensor，避免空列表时 .item() 报错
        loss_span = torch.tensor(0.0, device=device)
        num_matched = sum(len(q) for q, _ in indices)

        if num_matched > 0:
            seq_len = outputs["pred_cause_span"][0].shape[-1]
            positions = torch.arange(seq_len, device=device, dtype=torch.float)

            def calc_soft_length(s_logits, e_logits):
                prob_s, prob_e = F.softmax(s_logits, dim=-1), F.softmax(
                    e_logits, dim=-1
                )
                exp_s, exp_e = torch.sum(prob_s * positions), torch.sum(
                    prob_e * positions
                )
                return exp_e - exp_s

            for b, (q_idx, gt_idx) in enumerate(indices):
                if len(gt_idx) == 0:
                    continue
                tgt = targets[b]
                for q, gt in zip(q_idx, gt_idx):
                    # Cause
                    c_s, c_e = tgt["cause_spans"][gt]
                    loss_span += self.span_loss_fn(
                        outputs["pred_cause_span"][0][b, q].unsqueeze(0),
                        c_s.unsqueeze(0).to(device),
                    )
                    loss_span += self.span_loss_fn(
                        outputs["pred_cause_span"][1][b, q].unsqueeze(0),
                        c_e.unsqueeze(0).to(device),
                    )
                    loss_span += self.weight_len * F.smooth_l1_loss(
                        calc_soft_length(
                            outputs["pred_cause_span"][0][b, q],
                            outputs["pred_cause_span"][1][b, q],
                        ),
                        torch.tensor(float(c_e - c_s), device=device),
                    )

                    # Effect
                    e_s, e_e = tgt["effect_spans"][gt]
                    loss_span += self.span_loss_fn(
                        outputs["pred_effect_span"][0][b, q].unsqueeze(0),
                        e_s.unsqueeze(0).to(device),
                    )
                    loss_span += self.span_loss_fn(
                        outputs["pred_effect_span"][1][b, q].unsqueeze(0),
                        e_e.unsqueeze(0).to(device),
                    )
                    loss_span += self.weight_len * F.smooth_l1_loss(
                        calc_soft_length(
                            outputs["pred_effect_span"][0][b, q],
                            outputs["pred_effect_span"][1][b, q],
                        ),
                        torch.tensor(float(e_e - e_s), device=device),
                    )

                    # Trigger (显式)
                    if tgt["labels"][gt] == 0:
                        t_s, t_e = tgt["trigger_spans"][gt]
                        loss_span += self.span_loss_fn(
                            outputs["pred_trigger_span"][0][b, q].unsqueeze(0),
                            t_s.unsqueeze(0).to(device),
                        )
                        loss_span += self.span_loss_fn(
                            outputs["pred_trigger_span"][1][b, q].unsqueeze(0),
                            t_e.unsqueeze(0).to(device),
                        )

            loss_span = loss_span / num_matched

        loss_aux = self.aux_loss_fn(
            outputs["aux_exp_logits"], aux_labels[:, 0].to(device)
        ) + self.aux_loss_fn(outputs["aux_imp_logits"], aux_labels[:, 1].to(device))

        cos_sim = F.cosine_similarity(outputs["h_exp"], outputs["h_imp"], dim=-1)
        loss_ispd = torch.clamp(cos_sim - 0.1, min=0.0).mean()

        total_loss = (
            self.weight_cls * loss_cls
            + self.weight_span * loss_span
            + self.weight_aux * loss_aux
            + self.weight_ispd * loss_ispd
        )

        return total_loss, {
            "loss_cls": loss_cls.item(),
            "loss_span": loss_span.item(),
            "loss_aux": loss_aux.item(),
            "loss_ispd": loss_ispd.item(),
        }


# ==========================================
# 2. 训练器类
# ==========================================
class Trainer:
    def __init__(self, config, model, train_loader, val_loader=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # 1. 初始化损失函数
        self.criterion = SetCriterion(
            num_queries_exp=config.num_queries_exp,
            num_queries_imp=config.num_queries_imp,
            weight_cls=config.weight_cls,
            weight_span=config.weight_span,
            weight_len=config.weight_len,
            weight_aux=config.weight_aux,
            weight_ispd=config.weight_ispd,
        ).to(self.device)

        # 2. 分层学习率优化器
        encoder_params = list(self.model.encoder.parameters())
        decoder_head_params = [
            p for n, p in self.model.named_parameters() if not n.startswith("encoder.")
        ]

        self.optimizer = AdamW(
            [
                {"params": encoder_params, "lr": config.bert_learning_rate},
                {"params": decoder_head_params, "lr": config.learning_rate},
            ],
            weight_decay=config.weight_decay,
        )

        # 3. 学习率调度器
        total_steps = len(train_loader) * config.epochs
        warmup_steps = int(config.warm_factor * total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # 确保模型保存目录存在
        config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def train(self):
        print(f"\n[*] 开始训练，共 {self.config.epochs} 个 Epoch...")
        print(f"    Device: {self.device}")
        print(
            f"    BERT LR: {self.config.bert_learning_rate}, Head LR: {self.config.learning_rate}"
        )

        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss_accum = 0.0
            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"
            )

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                aux_labels = batch["aux_labels"].to(self.device)
                targets = batch["targets"]

                outputs = self.model(input_ids, attention_mask)
                loss, loss_dict = self.criterion(outputs, targets, aux_labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.clip_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                total_loss_accum += loss.item()
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Cls": f"{loss_dict['loss_cls']:.4f}",
                        "Span": f"{loss_dict['loss_span']:.4f}",
                        "LR": f"{self.optimizer.param_groups[1]['lr']:.2e}",  # 显示 Head 的学习率
                    }
                )

            avg_loss = total_loss_accum / len(self.train_loader)
            print(f"✅ Epoch {epoch+1} 结束! 平均 Loss: {avg_loss:.4f}")

            # 验证逻辑 (预留)
            if self.val_loader is not None:
                self._validate(epoch)

        # 训练结束后保存模型
        self._save_model()

    def _validate(self, epoch):
        """验证逻辑，可在此处补充评估指标计算"""
        # TODO: 实现验证集的评估逻辑
        pass

    def _save_model(self):
        """保存模型权重"""
        save_path = self.config.MODEL_SAVE_DIR / "causal_set_extractor.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"\n🎉 训练完成！模型已保存至: {save_path}")
