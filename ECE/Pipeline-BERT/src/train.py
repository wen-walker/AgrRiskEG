import torch
import torch.nn as nn
from tqdm import tqdm
import transformers

class Trainer(object):
    def __init__(self, model, updates_total, config):
        self.model = model
        self.updates_total = updates_total
        self.config = config
        # 参数分层，为BERT和非BERT参数设置不同的学习率和权重衰减
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]
        # 优化器
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        # 学习率调度器
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * self.updates_total,
                                                                      num_training_steps=self.updates_total)   

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", colour='cyan'):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                token_type_ids = batch['token_type_ids'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                outputs = self.model(input_ids, attention_mask, token_type_ids, labels)
                loss = outputs.loss
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Training", colour='cyan'):
            # 将输入数据移动到指定设备
            input_ids = batch['input_ids'].to(self.config.device) # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(self.config.device) # (batch_size, seq_len)
            token_type_ids = batch['token_type_ids'].to(self.config.device) # (batch_size, seq_len)
            labels = batch['labels'].to(self.config.device) # (batch_size, seq_len)
            # 清零梯度
            self.optimizer.zero_grad()
            # 前向传播
            outputs = self.model(input_ids, attention_mask, token_type_ids, labels) # (batch_size, seq_len, num_labels)
            # 计算损失
            # loss = self.criterion(outputs.contiguous().view(-1, len(self.config.labels)), labels.contiguous().view(-1))
            # loss = self.criterion(outputs.permute(0, 2, 1), labels)
            loss = outputs.loss
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()        
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def train(self, dataloaders):
        # 3.将模型移动到指定设备
        self.model.to(self.config.device)
        # 4.加载训练集和验证集
        train_loader, val_loader = dataloaders["train"], dataloaders["val"]
        # 5.开始训练
        best_val_loss = float("inf")
        patience = 5
        for epoch in range(1, self.config.epochs + 1):
            print(f"=============== Epoch {epoch}/{self.config.epochs} ==============")
            train_loss = self.train_one_epoch(train_loader)
            print(f"Training Loss: {train_loss:.4f}")
            val_loss = self.evaluate(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.save_path)
                print(f"Model saved to {self.config.save_path} with loss {best_val_loss:.4f}")
                patience = 5
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping triggered.")
                    break