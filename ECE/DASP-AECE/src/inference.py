import torch
from config import Config
from transformers import AutoTokenizer
from model import CausalSetExtractor


class CausalExtractorPipeline:
    """
    端到端农业因果抽取推理管道
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_span_len=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_span_len = max_span_len

        self.exp_id = tokenizer.convert_tokens_to_ids("[EXP]")
        self.imp_id = tokenizer.convert_tokens_to_ids("[IMP]")

    @staticmethod
    def _decode_predictions(outputs, max_span_len=30):
        """将模型输出解码为三元组集合"""
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

    def extract(self, text, max_length=256):
        # 1. 文本预处理与对齐
        encoding = self.tokenizer(
            text,
            max_length=max_length - 2,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"]

        new_input_ids = [input_ids[0], self.exp_id, self.imp_id] + input_ids[1:]
        new_attention_mask = [1, 1, 1] + attention_mask[1:]
        new_offsets = [(0, 0), (0, 0), (0, 0)] + offsets[1:]

        tensor_input_ids = torch.tensor([new_input_ids], dtype=torch.long).to(
            self.device
        )
        tensor_mask = torch.tensor([new_attention_mask], dtype=torch.long).to(
            self.device
        )

        # 2. 模型前向推理
        with torch.no_grad():
            outputs = self.model(tensor_input_ids, tensor_mask)

        pred_set = self._decode_predictions(outputs, self.max_span_len)[0]

        # 3. Token 到 Char 的逆向映射转换
        results = []

        def get_text_span_safe(source_text, offset_arr, start_tok, end_tok):
            if start_tok == -1 or end_tok == -1:
                return None
            if start_tok >= len(offset_arr) or end_tok >= len(offset_arr):
                return None

            char_start = offset_arr[start_tok][0]
            char_end = offset_arr[end_tok][1]

            if char_start == char_end == 0:
                return None
            return source_text[char_start:char_end]

        for pred in pred_set:
            cls_id, c_s, c_e, t_s, t_e, e_s, e_e = pred

            cause_text = get_text_span_safe(text, new_offsets, c_s, c_e)
            effect_text = get_text_span_safe(text, new_offsets, e_s, e_e)
            trigger_text = (
                get_text_span_safe(text, new_offsets, t_s, t_e) if cls_id == 0 else None
            )

            if not cause_text or not effect_text:
                continue

            rel_type = "Explicit (显式)" if cls_id == 0 else "Implicit (隐式)"

            results.append(
                {
                    "Type": rel_type,
                    "Cause": cause_text,
                    "Trigger": trigger_text,
                    "Effect": effect_text,
                }
            )

        return results


# ==========================================
# 独立测试区域
# ==========================================
if __name__ == "__main__":
    # 1. 加载配置
    config = Config()

    # 2. 加载 Tokenizer 并扩充词表
    print("[*] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.bert_name, cache_dir=config.CACHE_DIR
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["[EXP]", "[IMP]"]})
    config.tokenizer_len = len(tokenizer)

    # 3. 加载模型架构及权重
    print("[*] 加载模型...")
    model = CausalSetExtractor(config)

    model_path = config.MODEL_SAVE_DIR / "causal_set_extractor.pth"
    if not model_path.exists():
        print(f"⚠️ 未找到训练好的模型权重: {model_path}")
        print("⚠️ 请先运行 main.py 完成模型训练！")
        exit()

    model.load_state_dict(torch.load(model_path, map_location=config.device))
    print("✅ 模型权重加载成功！")

    # 4. 实例化推理管道
    pipeline = CausalExtractorPipeline(
        model, tokenizer, max_span_len=config.max_span_len, device=config.device
    )

    # 5. 测试用例
    test_sentences = [
        "连日暴雨造成玉米田大面积积水，玉米受涝严重；同时气温骤降，部分晚播玉米遭受低温冷害。",
        "棉花花铃期降水偏多，蕾铃脱落严重，同时土壤湿度较大。大风、暴雨灾害造成部分棉田出现积水现象。",
        "夏季高温导致玉米减产，早霜冻害使油菜受损严重。",
        "连日暴雨造成麦田大面积积水，小麦受涝严重；同时气温骤降，部分晚小麦受严重冻害。",
        "干旱持续，导致水稻生长受阻，产量下降；农民收入降低",
        "气孔导度通常会受叶片氮含量、光照强度、外界co2浓度、温度以及水分状况等环境因素的影响。当光强较低时，气孔导度下降，低光下的气孔导度会影响光合作用的诱导。在光合作用诱导过程中，气孔对光合作用的限制可以通过计算气孔下腔的CO2浓度的变化得到。在一定范围内的光照强度下气孔导度随着光强的增加而增加。在从低光向高光转化的过程中气孔导度逐渐上升。有研究表明气孔导度对光强变化的响应速率要慢于光合相关酶，气孔的完全开放需要至少30min。",
        "拔节期小麦遭遇持续干旱，土壤相对含水量降至30%以下。土壤相对含水量降至30%以下，植株水分亏缺。植株水分亏缺，细胞膨压显著降低，导致小麦茎秆节间伸长受阻，株高明显矮化。同时，叶片水分散失加剧，叶片水分散失加剧，叶绿素降解加快。叶绿素降解加快引起光合能力骤降，光合能力骤降使得干物质积累量锐减，干物质积累量锐减造成小麦千粒重下降。",
        "高温期喷施浓度过大的除草剂，黄瓜叶片产生药害斑点；同时底肥过量造成烧根，根系吸收受阻，植株萎蔫，雌花脱落，最终导致产量锐减。",
        "综合以上结果，施钙不仅极显著改善花生幼苗的表现型性状，提高相关的生理生化机能，而且大幅度提高花生抗逆基因的表达量，从而提高幼苗的耐渍涝性。",
    ]

    print("\n[*] 开始推理测试...")
    for i, text in enumerate(test_sentences):
        print(f"\n" + "=" * 50)
        print(f"📖 测试文本 {i+1}:\n{text}")
        print("-" * 50)

        extracted_results = pipeline.extract(text)

        if not extracted_results:
            print("🚫 未提取到任何因果关系。")
        else:
            print(f"✅ 成功提取到 {len(extracted_results)} 个因果对：\n")
            for j, res in enumerate(extracted_results):
                print(f"  [{j+1}] 类别: {res['Type']}")
                print(f"      🔹 原因:   {res['Cause']}")
                if res["Trigger"]:
                    print(f"      🔸 触发词:  {res['Trigger']}")
                print(f"      🔻 结果:  {res['Effect']}\n")
