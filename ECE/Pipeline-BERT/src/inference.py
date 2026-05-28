import torch
import sys
from transformers import AutoTokenizer
import argparse

# 引入你的配置和模型
from config import Config
from model import BIOBertCRFNer, BIOBertNer, RelationModel


class Predictor:
    def __init__(self, config):
        self.config = config
        # 修正：统一使用 config 中的 device
        self.device = config.device
        print(f"Loading models on {self.device}...")

        # --- 1. 加载 Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.bert_name, cache_dir=str(config.CACHE_DIR)
        )
        # 必须添加 RE 阶段用到的特殊符号，否则 RE 模型无法工作
        self.special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.special_tokens}
        )

        # --- 2. 加载 NER 模型 ---
        print(f"Loading NER Model ({config.model_name})...")

        # 根据配置实例化不同的模型类
        if config.model_name == "BIOBertNer":
            self.ner_model = BIOBertNer(config)
        elif config.model_name == "BIOBertCRFNer":
            self.ner_model = BIOBertCRFNer(config)
        else:
            raise ValueError(f"Unknown model name: {config.model_name}")

        ner_path = config.MODEL_SAVE_DIR / "model_en.pth"
        if not ner_path.exists():
            raise FileNotFoundError(f"NER model not found at {ner_path}")

        # 加载权重
        self.ner_model.load_state_dict(torch.load(ner_path, map_location=self.device))
        self.ner_model.to(self.device)
        self.ner_model.eval()

        # --- 3. 加载 RE 模型 ---
        print("Loading RE Model...")
        self.re_model = RelationModel(config, self.tokenizer)
        re_path = config.MODEL_SAVE_DIR / "model_re.pth"
        if not re_path.exists():
            raise FileNotFoundError(f"RE model not found at {re_path}")

        self.re_model.load_state_dict(torch.load(re_path, map_location=self.device))
        self.re_model.to(self.device)
        self.re_model.eval()

        # 构建 id2label 映射
        self.id2label = {v: k for k, v in config.labels.items()}

    def ner_predict(self, text):
        """
        第一阶段：实体识别
        """
        # 1. 预处理
        text_chars = list(text)
        # 截断处理：防止超过 BERT 最大长度 (预留 [CLS] 和 [SEP])
        max_seq_len = 510
        if len(text_chars) > max_seq_len:
            text_chars = text_chars[:max_seq_len]
            print(f"Warning: Text truncated to {max_seq_len} chars.")

        input_ids = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + text_chars + ["[SEP]"]
        )

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_tensor).to(self.device)
        token_type_ids = torch.zeros_like(input_tensor).to(self.device)

        # 2. 模型推理与解码 (关键修改部分)
        tags = []
        with torch.no_grad():
            outputs = self.ner_model(input_tensor, attention_mask, token_type_ids)
            logits = outputs.logits

            # --- 分情况解码 ---
            if isinstance(self.ner_model, BIOBertNer):
                # 情况 A: 普通 BERT (输出是 Logits 分数)
                # logits shape: [1, Seq_Len, Num_Labels]
                # 需要取 argmax 得到索引
                preds = torch.argmax(logits, dim=-1)  # [1, Seq_Len]
                tags = preds[0].cpu().numpy().tolist()

            elif isinstance(self.ner_model, BIOBertCRFNer):
                # 情况 B: BERT + CRF (输出通常已经是解码后的 Tags)
                # 取决于你的 forward 实现，通常 CRF.decode 返回 List[List[int]] 或 Tensor
                if isinstance(logits, torch.Tensor):
                    tags = logits[0].cpu().numpy().tolist()
                elif isinstance(logits, list):
                    tags = logits[0]  # batch size 为 1
                else:
                    raise TypeError(
                        f"Unexpected output type from CRF model: {type(logits)}"
                    )

        # 3. 解析 BIO 标签
        # input_ids 长度 = len(text) + 2 ([CLS], [SEP])
        # tags 长度通常与 input_ids 一致

        # 去掉 [CLS] 和 [SEP] 对应的标签
        valid_tags = tags[1:-1]

        # 安全检查：确保标签长度与文本长度一致
        current_text_len = len(text_chars)
        if len(valid_tags) > current_text_len:
            valid_tags = valid_tags[:current_text_len]
        elif len(valid_tags) < current_text_len:
            # 如果 CRF 解码长度不足（有时发生），补 O
            valid_tags += [self.config.labels["Other"]] * (
                current_text_len - len(valid_tags)
            )

        entities = []
        current_entity = None

        for i, tag_id in enumerate(valid_tags):
            label = self.id2label.get(tag_id, "Other")  # 默认 Other 防止 key error

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": text_chars[i],
                    "start": i,
                    "end": i + 1,
                    "type": label.split("-")[1],
                }
            elif label.startswith("I-"):
                if current_entity and label.split("-")[1] == current_entity["type"]:
                    current_entity["text"] += text_chars[i]
                    current_entity["end"] += 1
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            else:  # O or Other
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def re_predict(self, text, e1, e2):
        """
        第二阶段：关系判断
        """
        # 1. 插入标记 (使用 Entity Markers)
        markers = [
            {"pos": e1["start"], "token": "[E1]"},
            {"pos": e1["end"], "token": "[/E1]"},
            {"pos": e2["start"], "token": "[E2]"},
            {"pos": e2["end"], "token": "[/E2]"},
        ]
        markers.sort(key=lambda x: x["pos"], reverse=True)

        text_list = list(text)
        for m in markers:
            # 简单保护：防止索引越界 (虽然理论上 NER 出来的坐标是对的)
            if m["pos"] <= len(text_list):
                text_list.insert(m["pos"], m["token"])

        marked_text = "".join(text_list)

        # 2. Tokenize
        encoding = self.tokenizer(
            marked_text,
            max_length=self.config.max_len if hasattr(self.config, "max_len") else 256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        mask = encoding["attention_mask"].to(self.device)
        token_type = encoding["token_type_ids"].to(self.device)

        # 3. 预测
        with torch.no_grad():
            output = self.re_model(input_ids, mask, token_type)
            # RE 模型也是二分类 logits，需要 argmax
            logits = output.logits
            pred = torch.argmax(logits, dim=1).item()

        return pred == 1

    def run_pipeline(self, text):
        """执行完整流水线"""
        # Step 1: NER
        entities = self.ner_predict(text)

        # 过滤出 Event 和 Trigger
        events = [e for e in entities if "EVENT" in e["type"]]
        triggers = [e for e in entities if "TRIG" in e["type"]]

        print(
            f"🤖 -> 识别到实体: {[e['text'] + '(' + e['type'] + ')' for e in entities]}"
        )

        triples = []

        # Step 2: RE (两两配对)
        for i, e1 in enumerate(events):
            for j, e2 in enumerate(events):
                if i == j:
                    continue

                # 预测是否存在 e1 -> e2 的关系
                has_relation = self.re_predict(text, e1, e2)

                if has_relation:
                    # Step 3: 寻找中间的 Trigger (后处理)
                    found_trig = None
                    for trig in triggers:
                        # 逻辑：Trigger 必须在 e1 和 e2 之间
                        # 注意：这里假设文本顺序是 E1...Trigger...E2
                        # 如果是倒装句 (E2...Trigger...E1)，逻辑需要微调，或者直接取两位置中间的词
                        start_bound = min(e1["end"], e2["end"])
                        end_bound = max(e1["start"], e2["start"])

                        if trig["start"] >= start_bound and trig["end"] <= end_bound:
                            found_trig = trig["text"]
                            break

                    relation_word = found_trig if found_trig else "NULL"
                    triples.append(f"< {e1['text']} | {relation_word} | {e2['text']} >")

        return triples


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BIOBertNer", help="模型名称")
    parser.add_argument("--save_path", type=str, default=None, help="模型保存路径")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="训练设备")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率")
    parser.add_argument("--bert_name", type=str, default=None, help="预训练模型名称")
    args = parser.parse_args()

    # 初始化 Config
    conf = Config(args=args)

    try:
        predictor = Predictor(conf)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n初始化失败: {e}")
        return

    print("\n" + "=" * 50)
    print("🌾 农业气象灾害事件抽取交互系统 🌾")
    print(" 输入 'q' 或 'quit' 退出")
    print("=" * 50)

    while True:
        try:
            text = input("\n🧑 User >>: ").strip()
            if not text:
                continue
            if text.lower() in ["q", "quit", "exit"]:
                print("👋 Bye!")
                break

            triples = predictor.run_pipeline(text)

            print("抽取结果:")
            if triples:
                for t in triples:
                    print(t)
            else:
                print("未提取到有效三元组。")

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"处理出错: {e}")


if __name__ == "__main__":
    main()
