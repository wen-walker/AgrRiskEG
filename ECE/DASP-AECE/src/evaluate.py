import torch
import json
from pathlib import Path


class Evaluator:
    def __init__(self, config, model, tokenizer, test_loader, checkpoint_path=None):
        self.config = config
        self.device = config.device
        # 🌟 核心修改：如果传入了路径，则加载权重；否则直接使用传入的 model
        if checkpoint_path is not None:
            print(f"[*] 从硬盘加载模型权重: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.test_loader = test_loader

    # ==========================================
    # 1. 核心解码逻辑
    # ==========================================
    @staticmethod
    def _decode_predictions(outputs, max_span_len=30):
        bs = outputs["pred_logits"].shape[0]
        pred_classes = outputs["pred_logits"].argmax(dim=-1)

        c_s = outputs["pred_cause_span"][0].argmax(dim=-1)
        c_e = outputs["pred_cause_span"][1].argmax(dim=-1)
        t_s = outputs["pred_trigger_span"][0].argmax(dim=-1)
        t_e = outputs["pred_trigger_span"][1].argmax(dim=-1)
        e_s = outputs["pred_effect_span"][0].argmax(dim=-1)
        e_e = outputs["pred_effect_span"][1].argmax(dim=-1)

        batch_predictions = []
        for b in range(bs):
            pred_set = set()
            for q in range(outputs["pred_logits"].shape[1]):
                cls_id = pred_classes[b, q].item()
                if cls_id == 2:
                    continue

                cause = (c_s[b, q].item(), c_e[b, q].item())
                effect = (e_s[b, q].item(), e_e[b, q].item())

                if cause[0] > cause[1] or cause[1] - cause[0] > max_span_len:
                    continue
                if effect[0] > effect[1] or effect[1] - effect[0] > max_span_len:
                    continue

                if cls_id == 0:
                    trigger = (t_s[b, q].item(), t_e[b, q].item())
                    if trigger[0] > trigger[1] or trigger[1] - trigger[0] > 10:
                        continue
                else:
                    trigger = (-1, -1)

                if cause == effect:
                    continue

                pred_set.add(
                    (
                        cls_id,
                        cause[0],
                        cause[1],
                        trigger[0],
                        trigger[1],
                        effect[0],
                        effect[1],
                    )
                )
            batch_predictions.append(pred_set)
        return batch_predictions

    @staticmethod
    def _get_gt_sets(targets):
        batch_gts = []
        for tgt in targets:
            gt_set = set()
            for i in range(len(tgt["labels"])):
                cls_id = tgt["labels"][i].item()
                c_s, c_e = tgt["cause_spans"][i].tolist()
                t_s, t_e = tgt["trigger_spans"][i].tolist()
                e_s, e_e = tgt["effect_spans"][i].tolist()
                gt_set.add((cls_id, c_s, c_e, t_s, t_e, e_s, e_e))
            batch_gts.append(gt_set)
        return batch_gts

    # ==========================================
    # 2. 宽松匹配辅助函数
    # ==========================================
    @staticmethod
    def _is_span_overlap(span1, span2):
        if span1 == (-1, -1) and span2 == (-1, -1):
            return True
        if span1 == (-1, -1) or span2 == (-1, -1):
            return False
        return max(span1[0], span2[0]) <= min(span1[1], span2[1])

    @staticmethod
    def _tuple_match_relaxed(pred, gt):
        if pred[0] != gt[0]:
            return False
        if not Evaluator._is_span_overlap((pred[1], pred[2]), (gt[1], gt[2])):
            return False
        if not Evaluator._is_span_overlap((pred[5], pred[6]), (gt[5], gt[6])):
            return False
        return True

    # ==========================================
    # 3. 反向解码为人类可读文本
    # ==========================================
    @staticmethod
    def _ids_to_text(tokenizer, input_ids, start_idx, end_idx):
        if start_idx < 0 or end_idx < 0:
            return None
        tokens = input_ids[start_idx : end_idx + 1]
        text = tokenizer.decode(tokens, skip_special_tokens=True).replace(" ", "")
        return text if text else "[越界/无效为空]"

    # ==========================================
    # 4. 计算 P / R / F1
    # ==========================================
    @staticmethod
    def _calc_scores(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    # ==========================================
    # 5. Span 级别匹配
    # ==========================================
    @staticmethod
    def _update_span_metrics(span_metrics, pred_set, gt_set, role):
        role_slice = {
            "cause": (1, 2),
            "trigger": (3, 4),
            "effect": (5, 6),
        }
        s_idx, e_idx = role_slice[role]

        def extract_spans(tup_set):
            spans = set()
            for t in tup_set:
                if role == "trigger" and t[0] != 0:
                    continue
                spans.add((t[s_idx], t[e_idx]))
            return spans

        pred_spans = extract_spans(pred_set)
        gt_spans = extract_spans(gt_set)

        span_metrics[role]["tp"] += len(pred_spans & gt_spans)
        span_metrics[role]["fp"] += len(pred_spans - gt_spans)
        span_metrics[role]["fn"] += len(gt_spans - pred_spans)

    # ==========================================
    # 6. 核心评估接口
    # ==========================================
    def run_evaluate(self, output_bad_case_file=None):
        """
        执行评估逻辑，返回各项指标的 F1 分数字典
        """
        if output_bad_case_file is None:
            output_bad_case_file = self.config.OUTPUT_DIR / "bad_cases.jsonl"

        print("\n[*] 开始在测试集上进行评估，并导出错例分析报告...")

        # 确保模型在评估模式
        self.model.eval()

        def empty_counter():
            return {"tp": 0, "fp": 0, "fn": 0}

        triplet_metrics = {
            "exact": {
                "all": empty_counter(),
                "explicit": empty_counter(),
                "implicit": empty_counter(),
            },
            "relax": {
                "all": empty_counter(),
                "explicit": empty_counter(),
                "implicit": empty_counter(),
            },
        }

        span_metrics = {
            "cause": empty_counter(),
            "trigger": empty_counter(),
            "effect": empty_counter(),
        }

        bad_cases_count = 0

        with open(output_bad_case_file, "w", encoding="utf-8") as f_out:
            with torch.no_grad():
                for batch in self.test_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    outputs = self.model(input_ids, attention_mask)

                    pred_sets = self._decode_predictions(
                        outputs, max_span_len=self.config.max_span_len
                    )
                    gt_sets = self._get_gt_sets(batch["targets"])

                    for b in range(len(pred_sets)):
                        p_set = pred_sets[b]
                        g_set = gt_sets[b]
                        cur_input_ids = input_ids[b]

                        # 按类型拆分
                        p_explicit = {t for t in p_set if t[0] == 0}
                        p_implicit = {t for t in p_set if t[0] == 1}
                        g_explicit = {t for t in g_set if t[0] == 0}
                        g_implicit = {t for t in g_set if t[0] == 1}

                        # 严格匹配
                        for subset_key, p_sub, g_sub in [
                            ("all", p_set, g_set),
                            ("explicit", p_explicit, g_explicit),
                            ("implicit", p_implicit, g_implicit),
                        ]:
                            tp = len(p_sub & g_sub)
                            triplet_metrics["exact"][subset_key]["tp"] += tp
                            triplet_metrics["exact"][subset_key]["fp"] += (
                                len(p_sub) - tp
                            )
                            triplet_metrics["exact"][subset_key]["fn"] += (
                                len(g_sub) - tp
                            )

                        # 宽松匹配
                        for subset_key, p_sub, g_sub in [
                            ("all", p_set, g_set),
                            ("explicit", p_explicit, g_explicit),
                            ("implicit", p_implicit, g_implicit),
                        ]:
                            p_list, g_list = list(p_sub), list(g_sub)
                            matched_p, matched_g = set(), set()
                            for g_idx, gt in enumerate(g_list):
                                for p_idx, pred in enumerate(p_list):
                                    if (
                                        p_idx not in matched_p
                                        and self._tuple_match_relaxed(pred, gt)
                                    ):
                                        triplet_metrics["relax"][subset_key]["tp"] += 1
                                        matched_p.add(p_idx)
                                        matched_g.add(g_idx)
                                        break
                            triplet_metrics["relax"][subset_key]["fp"] += len(
                                p_list
                            ) - len(matched_p)
                            triplet_metrics["relax"][subset_key]["fn"] += len(
                                g_list
                            ) - len(matched_g)

                        # Span 指标
                        for role in ("cause", "trigger", "effect"):
                            self._update_span_metrics(span_metrics, p_set, g_set, role)

                        # Bad Case 记录
                        if p_set != g_set:
                            bad_cases_count += 1
                            raw_text = self.tokenizer.decode(
                                cur_input_ids, skip_special_tokens=True
                            ).replace(" ", "")

                            def make_readable(tup_set):
                                result = []
                                for t in sorted(tup_set):
                                    cls_id, c_s, c_e, t_s, t_e, e_s, e_e = t
                                    result.append(
                                        {
                                            "type": (
                                                "Explicit"
                                                if cls_id == 0
                                                else "Implicit"
                                            ),
                                            "cause": self._ids_to_text(
                                                self.tokenizer, cur_input_ids, c_s, c_e
                                            ),
                                            "trigger": self._ids_to_text(
                                                self.tokenizer, cur_input_ids, t_s, t_e
                                            ),
                                            "effect": self._ids_to_text(
                                                self.tokenizer, cur_input_ids, e_s, e_e
                                            ),
                                        }
                                    )
                                return result

                            f_out.write(
                                json.dumps(
                                    {
                                        "text": raw_text,
                                        "ground_truth": make_readable(g_set),
                                        "prediction": make_readable(p_set),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

        # ==========================================
        # 打印结果
        # ==========================================
        W = 70
        print("=" * W)
        print(" 评估结果总览 ".center(W))

        print("=" * W)
        print(f"  {'三元组指标'}")
        print("-" * W)
        # header = f"  {'类型':<14} {'匹配方式':<10} {'TP':>6} {'FP':>6} {'FN':>6}  {'P':>7} {'R':>7} {'F1':>7}"
        header = f"  {'类型':<14} {'匹配方式':<7} {'P':>7} {'R':>7} {'F1':>7}"
        print(header)
        print("-" * W)

        label_map = {"all": "全部", "explicit": "显式", "implicit": "隐式"}
        final_scores = {}
        for match_type in ("exact", "relax"):
            match_label = "严格匹配" if match_type == "exact" else "宽松匹配"
            for subset_key in ("all", "explicit", "implicit"):
                c = triplet_metrics[match_type][subset_key]
                p, r, f = self._calc_scores(c["tp"], c["fp"], c["fn"])
                final_scores[f"{match_type}_{subset_key}_f1"] = f
                print(
                    f"  {label_map[subset_key]:<14} {match_label:<10} "
                    # f"{c['tp']:>6} {c['fp']:>6} {c['fn']:>6}  "
                    f"{p:>7.4f} {r:>7.4f} {f:>7.4f}"
                )
            print("-" * W)

        print(f"  {'Span 级别指标'}")
        print("-" * W)
        # print(
        #     f"  {'角色':<16} {'TP':>6} {'FP':>6} {'FN':>6}  {'P':>7} {'R':>7} {'F1':>7}"
        # )
        print(
            f"  {'角色':<12} {'P':>7} {'R':>7} {'F1':>7}"
        )
        print("-" * W)
        role_label = {"cause": "原因", "trigger": "触发", "effect": "结果"}
        for role in ("cause", "trigger", "effect"):
            c = span_metrics[role]
            p, r, f = self._calc_scores(c["tp"], c["fp"], c["fn"])
            final_scores[f"span_{role}_f1"] = f
            # print(
            #     f"  {role_label[role]:<16} {c['tp']:>6} {c['fp']:>6} {c['fn']:>6}  "
            #     f"{p:>7.4f} {r:>7.4f} {f:>7.4f}"
            # )
            print(
                f"  {role_label[role]:<16}"
                f"{p:>7.4f} {r:>7.4f} {f:>7.4f}"
            )

        print("=" * W)
        print(f"  📁 已将 {bad_cases_count} 条错例导出至: {output_bad_case_file}")
        print("=" * W)

        # 评估结束，恢复训练模式
        self.model.train()

        return final_scores
