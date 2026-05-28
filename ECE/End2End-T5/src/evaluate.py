import torch
import json
import re
from tqdm import tqdm
from collections import defaultdict
import logging

# 引入语义相似度计算库
from sentence_transformers import SentenceTransformer, util

# 设置日志，避免 print 刷屏
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config, model, dataloader):
        """
        初始化评估器 - 适配三元组抽取任务
        :param config: 配置参数
        :param model: T5EEModel 实例
        :param dataloader: 验证集或测试集的 DataLoader
        """
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = model.tokenizer
        self.device = next(model.parameters()).device

        # --- 新增：加载语义匹配模型 ---
        print("🔄 正在加载语义评估模型 (text2vec-base-chinese)...")
        try:
            # 这是一个轻量且效果好的中文语义模型
            self.sim_model = SentenceTransformer(
                "shibing624/text2vec-base-chinese", cache_folder=config.CACHE_DIR
            )
            self.sim_threshold = 0.8  # 语义相似度阈值 (建议 0.8~0.9)
            print("✅ 语义模型加载完成")
        except Exception as e:
            print(f"❌ 语义模型加载失败: {e}")
            print("⚠️ 将回退到仅使用字符串匹配模式")
            self.sim_model = None

    def parse_text_to_triples(self, text):
        """
        核心解析函数：解析新格式 <E1|T|E2> 或 <E1|E2>
        输入: "<越冬期干旱|造成|小麦减产> <高温|玉米绝收>"
        输出: {("越冬期干旱", "造成", "小麦减产"), ("高温", "", "玉米绝收")}
        """
        triples = set()
        text = text.strip()
        if text == "无" or not text:
            return triples

        # 使用正则提取所有 <> 包含的内容
        pattern = r"<([^>]+)>"
        matches = re.findall(pattern, text)

        for content in matches:
            # 按 | 分割
            parts = content.split("|")

            # --- 修改点开始：支持 <A|B> 这种只有两部分的格式 ---

            # 情况 1: 标准完整格式 <前件|触发词|后件> (3部分)
            if len(parts) == 3:
                s = parts[0].strip()
                p = parts[1].strip()  # 触发词
                o = parts[2].strip()
                if s or o:
                    triples.add((s, p, o))

            # 情况 2: 省略触发词格式 <前件|后件> (2部分)
            # 例如预测为 <早霜冻害|油菜受损>，视为 (早霜冻害, "", 油菜受损)
            elif len(parts) == 2:
                s = parts[0].strip()
                o = parts[1].strip()
                if s or o:
                    triples.add((s, "", o))

            # --- 修改点结束 ---
            else:
                pass

        return triples

    def is_entity_match(self, pred_e, target_e):
        """
        混合匹配策略：
        1. 精确匹配 (最快)
        2. 字符重叠率 (较快)
        3. 语义相似度 (最慢但最准，兜底)
        """
        p = pred_e.replace(" ", "")
        t = target_e.replace(" ", "")

        # 1. 判空与精确匹配
        if not p or not t:
            return False
        if p == t:
            return True

        # 2. 字符重叠 (保留这个是为了处理包含关系，比如 "小麦" vs "冬小麦")
        if p in t or t in p:
            ratio = min(len(p), len(t)) / max(len(p), len(t))
            if ratio > 0.6:  # 字符重叠阈值
                return True

        # 3. 语义相似度匹配 (Semantic Similarity)
        if self.sim_model is not None:
            # 编码为向量
            emb1 = self.sim_model.encode(
                p, convert_to_tensor=True, show_progress_bar=False
            )
            emb2 = self.sim_model.encode(
                t, convert_to_tensor=True, show_progress_bar=False
            )

            # 计算余弦相似度
            cosine_score = util.cos_sim(emb1, emb2).item()

            if cosine_score >= self.sim_threshold:
                return True

        return False

    def is_trigger_match(self, pred_t, target_t):
        """
        软匹配辅助函数：判断触发词是否兼容
        """
        # 定义通用触发词表
        common_triggers = [
            "导致",
            "造成",
            "使",
            "影响",
            "引起",
            "促进",
            "抑制",
            "产生",
            "诱发",
            "致使",
            "意味着",
            "包括",
            "使得",
            "作为",
            "有关",
            "提供",
            "增加",
            "降低",
            "减少",
            "提高",
        ]

        p = pred_t.strip()
        t = target_t.strip()

        if p == t:
            return True

        # 情况A：标签是空的(隐式)，但模型预测了通用词(显式) -> 算对
        if t == "" and p in common_triggers:
            return True

        # 情况B：标签有通用词，模型没预测出来(为空) -> 算对
        if p == "" and t in common_triggers:
            return True

        # 其他情况按实体逻辑匹配 (防止触发词是比较复杂的短语，也走语义匹配)
        return self.is_entity_match(p, t)

    def calculate_metrics(self, true_count, pred_count, correct_count):
        """
        计算 Precision, Recall, F1
        """
        p = correct_count / pred_count if pred_count > 0 else 0.0
        r = correct_count / true_count if true_count > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def evaluate(self):
        """
        执行评估循环 - 输出多维度指标：
          1. 实体级别：原因 / 触发词 / 结果 各自的 P/R/F1
          2. 三元组级别：显性（触发词非空）/ 隐性（触发词为空）/ 整体 的 P/R/F1
        """
        self.model.eval()

        # ---- 实体级别计数器 ----
        ent_cause_true, ent_cause_pred, ent_cause_correct = 0, 0, 0
        ent_trig_true, ent_trig_pred, ent_trig_correct = 0, 0, 0
        ent_eff_true, ent_eff_pred, ent_eff_correct = 0, 0, 0

        # ---- 三元组级别计数器（按显性/隐性分类）----
        # 整体
        trip_all_true, trip_all_pred, trip_all_correct = 0, 0, 0
        # 显性（触发词非空）
        trip_exp_true, trip_exp_pred, trip_exp_correct = 0, 0, 0
        # 隐性（触发词为空）
        trip_imp_true, trip_imp_pred, trip_imp_correct = 0, 0, 0

        results_list = []

        with torch.no_grad():
            for batch in tqdm(
                self.dataloader,
                desc="Evaluating (Semantic)",
                leave=False,
                colour="green",
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 1. 模型生成
                generated_ids = self.model.predict(input_ids, attention_mask)

                # 2. 解码预测
                pred_texts = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # 3. 解码标签
                labels_copy = labels.clone()
                labels_copy[labels_copy == -100] = self.tokenizer.pad_token_id
                target_texts = self.tokenizer.batch_decode(
                    labels_copy, skip_special_tokens=True
                )

                # 4. 逐条对比
                for pred_text, target_text in zip(pred_texts, target_texts):
                    pred_set = self.parse_text_to_triples(pred_text)
                    target_set = self.parse_text_to_triples(target_text)

                    pred_list = list(pred_set)
                    target_list = list(target_set)

                    # ---- 三元组级别 Soft Match ----
                    current_correct = 0
                    matched_p_indices = set()
                    matched_t_indices = set()

                    for pi, p in enumerate(pred_list):
                        for ti, t in enumerate(target_list):
                            if ti in matched_t_indices:
                                continue
                            match_s = self.is_entity_match(p[0], t[0])
                            match_p = self.is_trigger_match(p[1], t[1])
                            match_o = self.is_entity_match(p[2], t[2])
                            if match_s and match_p and match_o:
                                current_correct += 1
                                matched_p_indices.add(pi)
                                matched_t_indices.add(ti)
                                break

                    # ---- 三元组级别：按显性/隐性分类计数 ----
                    # 整体
                    trip_all_pred += len(pred_list)
                    trip_all_true += len(target_list)
                    trip_all_correct += current_correct

                    # 显性（按 target 触发词分类；未匹配的 pred 按自身触发词分类）
                    for ti, t in enumerate(target_list):
                        if t[1].strip():
                            trip_exp_true += 1
                    for pi, p in enumerate(pred_list):
                        if p[1].strip():
                            trip_exp_pred += 1
                    for pi, ti in zip(matched_p_indices, matched_t_indices):
                        if target_list[ti][1].strip():
                            trip_exp_correct += 1

                    # 隐性（触发词为空）
                    for ti, t in enumerate(target_list):
                        if not t[1].strip():
                            trip_imp_true += 1
                    for pi, p in enumerate(pred_list):
                        if not p[1].strip():
                            trip_imp_pred += 1
                    for pi, ti in zip(matched_p_indices, matched_t_indices):
                        if not target_list[ti][1].strip():
                            trip_imp_correct += 1

                    # ---- 实体级别计数（独立匹配，不受三元组匹配约束）----
                    # 原因 (Cause)
                    used_t_cause = set()
                    for pi, p in enumerate(pred_list):
                        ent_cause_pred += 1
                        for ti, t in enumerate(target_list):
                            if ti not in used_t_cause and self.is_entity_match(
                                p[0], t[0]
                            ):
                                ent_cause_correct += 1
                                used_t_cause.add(ti)
                                break
                    ent_cause_true += len(target_list)

                    # 触发词 (Trigger)
                    used_t_trig = set()
                    for pi, p in enumerate(pred_list):
                        ent_trig_pred += 1
                        for ti, t in enumerate(target_list):
                            if ti not in used_t_trig and self.is_trigger_match(
                                p[1], t[1]
                            ):
                                ent_trig_correct += 1
                                used_t_trig.add(ti)
                                break
                    ent_trig_true += len(target_list)

                    # 结果 (Effect)
                    used_t_eff = set()
                    for pi, p in enumerate(pred_list):
                        ent_eff_pred += 1
                        for ti, t in enumerate(target_list):
                            if ti not in used_t_eff and self.is_entity_match(
                                p[2], t[2]
                            ):
                                ent_eff_correct += 1
                                used_t_eff.add(ti)
                                break
                    ent_eff_true += len(target_list)

                    # ---- 收集 Bad Cases ----
                    if pred_set != target_set:
                        results_list.append(
                            {
                                "target_raw": target_text,
                                "pred_raw": pred_text,
                                "target_parsed": list(target_set),
                                "pred_parsed": list(pred_list),
                                "missing_strict": list(target_set - pred_set),
                                "extra_strict": list(pred_set - target_set),
                            }
                        )

        # 5. 计算所有指标
        cause_p, cause_r, cause_f1 = self.calculate_metrics(
            ent_cause_true, ent_cause_pred, ent_cause_correct
        )
        trig_p, trig_r, trig_f1 = self.calculate_metrics(
            ent_trig_true, ent_trig_pred, ent_trig_correct
        )
        eff_p, eff_r, eff_f1 = self.calculate_metrics(
            ent_eff_true, ent_eff_pred, ent_eff_correct
        )

        exp_p, exp_r, exp_f1 = self.calculate_metrics(
            trip_exp_true, trip_exp_pred, trip_exp_correct
        )
        imp_p, imp_r, imp_f1 = self.calculate_metrics(
            trip_imp_true, trip_imp_pred, trip_imp_correct
        )
        all_p, all_r, all_f1 = self.calculate_metrics(
            trip_all_true, trip_all_pred, trip_all_correct
        )

        metrics = {
            "entity": {
                "cause":   {"precision": cause_p, "recall": cause_r, "f1": cause_f1},
                "trigger": {"precision": trig_p,  "recall": trig_r,  "f1": trig_f1},
                "effect":  {"precision": eff_p,   "recall": eff_r,   "f1": eff_f1},
            },
            "triplet": {
                "explicit": {"precision": exp_p, "recall": exp_r, "f1": exp_f1},
                "implicit": {"precision": imp_p, "recall": imp_r, "f1": imp_f1},
                "overall":  {"precision": all_p, "recall": all_r, "f1": all_f1},
            },
        }

        # 6. 打印报告
        semantic_info = (
            f"Threshold: {self.sim_threshold}" if self.sim_model else "Disabled"
        )
        print("\n" + "=" * 65)
        print(
            f"📊 EVALUATION REPORT (Semantic Soft Match | {semantic_info})"
        )

        print("\n【实体级别指标】")
        print(f"  {'类型':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("  " + "-" * 44)
        print(
            f"  {'原因':<8} {cause_p:>10.2%} {cause_r:>10.2%} {cause_f1:>10.2%}"
        )
        print(
            f"  {'触发':<8} {trig_p:>10.2%} {trig_r:>10.2%} {trig_f1:>10.2%}"
        )
        print(
            f"  {'结果':<8} {eff_p:>10.2%} {eff_r:>10.2%} {eff_f1:>10.2%}"
        )

        print("\n【三元组级别指标（显性 / 隐性 / 整体）】")
        print(f"  {'类型':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("  " + "-" * 44)
        print(
            f"  {'显性':<8} {exp_p:>10.2%} {exp_r:>10.2%} {exp_f1:>10.2%}"
        )
        print(
            f"  {'隐性':<8} {imp_p:>10.2%} {imp_r:>10.2%} {imp_f1:>10.2%}"
        )
        print(
            f"  {'整体':<8} {all_p:>10.2%} {all_r:>10.2%} {all_f1:>10.2%}"
        )
        print("=" * 65 + "\n")

        # 7. 保存 Bad Cases
        self.save_bad_cases(results_list)

        return metrics

    def save_bad_cases(self, results):
        if not results:
            return
        output_dir = self.config.PROCESSED_DATA_DIR.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "eval_bad_cases.json"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"📝 已将 {len(results)} 条差异样本保存至: {output_path}")
        except Exception as e:
            print(f"❌ 保存 Bad Cases 失败: {e}")
