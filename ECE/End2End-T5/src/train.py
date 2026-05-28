import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import os
import numpy as np


class Trainer:
    def __init__(self, config, model, train_loader, val_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 1. 设置设备 (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # 2. 优化器 (AdamW)
        # 过滤掉不需要权重衰减的参数 (如 LayerNorm 和 Bias)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

        # 3. 学习率调度器 (Linear Warmup)
        num_update_steps_per_epoch = len(train_loader)
        num_training_steps = config.epochs * num_update_steps_per_epoch
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(config.warm_factor * num_training_steps),
            num_training_steps=num_training_steps,
        )

        # 4. 混合精度训练 (FP16) - 节省显存
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None

        # 用于保存最佳模型
        self.best_loss = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]",
            leave=False,
            colour="cyan",
        )

        for batch in progress_bar:
            # 将数据移动到设备
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # --- 混合精度前向传播 ---
            # 如果 config.fp16 为 False，autocast 实际上什么都不做，兼容性很好
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                loss, _ = self.model(input_ids, attention_mask, labels)

            # --- 反向传播与优化 ---
            self.optimizer.zero_grad()

            if self.config.fp16:
                self.scaler.scale(loss).backward()
                # 梯度裁剪 (防止 T5 梯度爆炸)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad_norm
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad_norm
                )
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self, epoch):
        """
        验证集评估 - 计算 Validation Loss
        注意：这里暂不进行 generate 和 F1 计算（比较耗时），主要用于监控模型是否过拟合
        """
        self.model.eval()
        total_loss = 0

        if self.val_loader is None:
            return 0.0

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1} [Valid]",
                leave=False,
                colour="magenta",
            )
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                loss, _ = self.model(input_ids, attention_mask, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        print(f"🚀 开始训练... 设备: {self.device}")

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)

            # 打印日志
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_path = self.config.MODEL_SAVE_DIR / "best_model"
                print(
                    f"✨ 发现新最佳模型 (Loss: {val_loss:.4f})，正在保存至 {save_path} ..."
                )

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # 调用 model 中我们自定义的 save_pretrained
                self.model.save_pretrained(save_path)
            # (可选) 可以在这里添加 Early Stopping 逻辑

        print("✅ 训练结束！")
