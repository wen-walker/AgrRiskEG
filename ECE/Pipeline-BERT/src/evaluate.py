import torch
from tqdm import tqdm
from predict import predict_batch

class EntityEvaluator(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # 根据 config.labels 动态构建各类别的标签索引集合
        self.entity_groups = {
            "原因": set(),
            "触发": set(),
            "结果": set(),
        }
        for label_name, idx in config.labels.items():
            if "EVENT1" in label_name:
                self.entity_groups["原因"].add(idx)
            elif "EVENT2" in label_name:
                self.entity_groups["结果"].add(idx)
            elif "TRIG" in label_name:
                self.entity_groups["触发"].add(idx)

        # 每个类别的 TP / FP / FN 计数器
        self.counts = {name: {"TP": 0, "FP": 0, "FN": 0} for name in self.entity_groups}

    @staticmethod
    def _calc_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @staticmethod
    def _display_width(s):
        """计算字符串的终端显示宽度（中文字符按 2 计）"""
        return sum(2 if ord(c) > 127 else 1 for c in s)

    @classmethod
    def _pad(cls, s, width):
        """按显示宽度左对齐填充"""
        return s + ' ' * (width - cls._display_width(s))

    def evaluate_model(self, test_loader, model, config):
        # 每次评估前重置计数器
        for name in self.counts:
            self.counts[name] = {"TP": 0, "FP": 0, "FN": 0}

        for batch in tqdm(test_loader, desc="Evaluating", colour='cyan'):
            input_ids      = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            token_type_ids = batch['token_type_ids'].to(config.device)
            labels         = batch['labels'].to(config.device)
            preds          = predict_batch(input_ids, attention_mask, token_type_ids, model, config)
            preds          = torch.tensor(preds).to(config.device)

            # 展平
            preds_flat  = preds.contiguous().view(-1)
            labels_flat = labels.contiguous().view(-1)

            # 过滤掉标签为 -100 的位置
            mask        = labels_flat != -100
            preds_flat  = preds_flat[mask]
            labels_flat = labels_flat[mask]

            # 对每个类别分别统计 TP / FP / FN
            for name, label_ids in self.entity_groups.items():
                ids_tensor = torch.tensor(list(label_ids), device=config.device)
                is_true = torch.isin(labels_flat, ids_tensor)
                is_pred = torch.isin(preds_flat,  ids_tensor)

                self.counts[name]["TP"] += (is_true &  is_pred).sum().item()
                self.counts[name]["FP"] += (~is_true &  is_pred).sum().item()
                self.counts[name]["FN"] += (is_true & ~is_pred).sum().item()

        # ---------- 打印结果 ----------
        W = 8  # 实体名称列显示宽度
        sep  = f"+-{'─'*W} -+-----------+-----------+-----------+"
        hdr  = f"| {self._pad('实体', W)} | Precision | Recall    | F1-Score  |"

        print("\n===== Entity Evaluation Results =====")
        print(sep)
        print(hdr)
        print(sep)

        for name, counts in self.counts.items():
            p, r, f1 = self._calc_prf(counts["TP"], counts["FP"], counts["FN"])
            print(f"| {self._pad(name, W)} | {p:<9.4f} | {r:<9.4f} | {f1:<9.4f} |")

        # 整体 micro 平均
        total_TP = sum(c["TP"] for c in self.counts.values())
        total_FP = sum(c["FP"] for c in self.counts.values())
        total_FN = sum(c["FN"] for c in self.counts.values())
        p_all, r_all, f1_all = self._calc_prf(total_TP, total_FP, total_FN)
        print(sep)
        print(f"| {self._pad('Overall', W)} | {p_all:<9.4f} | {r_all:<9.4f} | {f1_all:<9.4f} |")
        print(sep)
        print("======================================")

    def run_evaluate(self, dataloaders):
        # 1.加载模型参数
        self.model.load_state_dict(torch.load(self.config.save_path, map_location=self.config.device))
        self.model.to(self.config.device)
        # 2.加载测试集
        test_loader = dataloaders["test"]
        self.evaluate_model(test_loader, self.model, self.config)





class RelationEvaluator(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        # 显式（含触发词）和隐式（无触发词）分别计数
        self.counts = {
            "显式": {"TP": 0, "FP": 0, "FN": 0},
            "隐式": {"TP": 0, "FP": 0, "FN": 0},
        }
        # 整体二分类计数（不区分类型）
        self.overall = {"TP": 0, "FP": 0, "FN": 0}

    @staticmethod
    def _calc_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @staticmethod
    def _display_width(s):
        """计算字符串的终端显示宽度（中文字符按 2 计）"""
        return sum(2 if ord(c) > 127 else 1 for c in s)

    @classmethod
    def _pad(cls, s, width):
        """按显示宽度左对齐填充"""
        return s + ' ' * (width - cls._display_width(s))

    def evaluate_model(self, test_loader, model, config):
        model.eval()

        # 每次评估前重置计数器
        for name in self.counts:
            self.counts[name] = {"TP": 0, "FP": 0, "FN": 0}
        self.overall = {"TP": 0, "FP": 0, "FN": 0}

        for batch in tqdm(test_loader, desc="Evaluating RE", colour="cyan"):
            input_ids      = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            token_type_ids = batch["token_type_ids"].to(config.device)
            labels         = batch["labels"].to(config.device)
            has_trig       = batch["has_trig"].to(config.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask, token_type_ids)
                logits  = outputs.logits
                preds   = torch.argmax(logits, dim=1)  # [batch_size]

            pred_pos   = preds  == 1
            label_pos  = labels == 1
            neg_mask   = ~label_pos   # labels == 0，对显式和隐式共用

            # ---- 整体二分类（不区分类型）----
            self.overall["TP"] += (label_pos  & pred_pos).sum().item()
            self.overall["FP"] += (neg_mask   & pred_pos).sum().item()
            self.overall["FN"] += (label_pos  & ~pred_pos).sum().item()

            # ---- 显式：gold positive class = label==1 AND has_trig==True ----
            explicit_pos = label_pos & has_trig
            self.counts["显式"]["TP"] += (explicit_pos &  pred_pos).sum().item()
            self.counts["显式"]["FP"] += (neg_mask     &  pred_pos).sum().item()
            self.counts["显式"]["FN"] += (explicit_pos & ~pred_pos).sum().item()

            # ---- 隐式：gold positive class = label==1 AND has_trig==False ----
            implicit_pos = label_pos & ~has_trig
            self.counts["隐式"]["TP"] += (implicit_pos &  pred_pos).sum().item()
            self.counts["隐式"]["FP"] += (neg_mask     &  pred_pos).sum().item()
            self.counts["隐式"]["FN"] += (implicit_pos & ~pred_pos).sum().item()

        # ---------- 打印结果 ----------
        W   = 8
        sep = f"+-{'─'*W} -+-----------+-----------+-----------+"
        hdr = f"| {self._pad('类型', W)} | Precision | Recall    | F1-Score  |"

        print("\n===== RE Evaluation Results =====")
        print(sep)
        print(hdr)
        print(sep)

        for name, counts in self.counts.items():
            p, r, f1 = self._calc_prf(counts["TP"], counts["FP"], counts["FN"])
            print(f"| {self._pad(name, W)} | {p:<9.4f} | {r:<9.4f} | {f1:<9.4f} |")

        # 整体（不区分类型）
        p_all, r_all, f1_all = self._calc_prf(
            self.overall["TP"], self.overall["FP"], self.overall["FN"]
        )
        print(sep)
        print(f"| {self._pad('Overall', W)} | {p_all:<9.4f} | {r_all:<9.4f} | {f1_all:<9.4f} |")
        print(sep)
        print("================================")

        return f1_all  # 返回整体 F1 以便外部记录 best_score

    def run_evaluate(self, dataloaders):
        print(f"Loading model from {self.config.save_path}...")
        self.model.load_state_dict(
            torch.load(self.config.save_path, map_location=self.config.device)
        )
        self.model.to(self.config.device)

        test_loader = dataloaders["test"]
        return self.evaluate_model(test_loader, self.model, self.config)