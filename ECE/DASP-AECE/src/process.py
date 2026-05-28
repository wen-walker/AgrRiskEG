import json
from pathlib import Path
from collections import defaultdict


class Processor(object):
    def __init__(self, config):
        self.config = config
        # 确保处理后的数据目录存在
        self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_label_studio_sample(sample):
        """
        解析单个 Label Studio 样本，提取显式和隐式因果四元组
        """
        text = sample.get("data", {}).get("text", "")
        sample_id = sample.get("id", "unknown")

        annotations = sample.get("annotations", [])
        if not annotations or annotations[0].get("was_cancelled", False):
            return None

        results = annotations[0].get("result", [])

        entities = {}
        directed_graph = defaultdict(set)  # 🌟 使用有向图存储关系

        for item in results:
            if item["type"] == "labels":
                ent_id = item["id"]
                ent_label = item["value"]["labels"][0]
                entities[ent_id] = {
                    "type": ent_label,
                    "start": item["value"]["start"],
                    "end": item["value"]["end"],
                    "text": item["value"]["text"],
                }
            elif item["type"] == "relation":
                # 🌟 严格只存正向关系，不抹平方向
                from_id = item["from_id"]
                to_id = item["to_id"]
                directed_graph[from_id].add(to_id)

        event1_ids = [eid for eid, data in entities.items() if data["type"] == "EVENT1"]

        causal_tuples = []
        has_exp = 0
        has_imp = 0

        # 🌟 核心寻路逻辑 (有向路径寻路)
        for e1_id in event1_ids:
            # 规则1：判断显式 - 路径必须是 E1 -> TRIG -> E2
            for neighbor_id in directed_graph[e1_id]:
                if entities[neighbor_id]["type"] == "TRIG":
                    trigger_id = neighbor_id
                    for e2_id in directed_graph[trigger_id]:  # TRIG 再次出发寻找 E2
                        if entities[e2_id]["type"] == "EVENT2":
                            has_exp = 1
                            causal_tuples.append(
                                {
                                    "type": "Explicit",
                                    "cause": [
                                        entities[e1_id]["start"],
                                        entities[e1_id]["end"],
                                    ],
                                    "trigger": [
                                        entities[trigger_id]["start"],
                                        entities[trigger_id]["end"],
                                    ],
                                    "effect": [
                                        entities[e2_id]["start"],
                                        entities[e2_id]["end"],
                                    ],
                                    "meta": {
                                        "cause_text": entities[e1_id]["text"],
                                        "trigger_text": entities[trigger_id]["text"],
                                        "effect_text": entities[e2_id]["text"],
                                    },
                                }
                            )

            # 规则2：判断隐式 - 路径必须是 E1 -> E2
            for neighbor_id in directed_graph[e1_id]:
                if entities[neighbor_id]["type"] == "EVENT2":
                    has_imp = 1
                    causal_tuples.append(
                        {
                            "type": "Implicit",
                            "cause": [entities[e1_id]["start"], entities[e1_id]["end"]],
                            "trigger": None,
                            "effect": [
                                entities[neighbor_id]["start"],
                                entities[neighbor_id]["end"],
                            ],
                            "meta": {
                                "cause_text": entities[e1_id]["text"],
                                "trigger_text": None,
                                "effect_text": entities[neighbor_id]["text"],
                            },
                        }
                    )

        return {
            "id": sample_id,
            "text": text,
            "causal_tuples": causal_tuples,
            "has_exp": has_exp,
            "has_imp": has_imp,
        }

    def _process_file(self, input_filename, output_filename):
        """处理单个文件并导出为 JSONL"""
        input_path = self.config.RAW_DATA_DIR / input_filename
        output_path = self.config.PROCESSED_DATA_DIR / output_filename

        if not input_path.exists():
            print(f"⚠️ 文件不存在跳过: {input_path}")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_count = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            for sample in data:
                processed_data = self._parse_label_studio_sample(sample)
                if processed_data is not None:
                    out_f.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
                    processed_count += 1

        print(
            f"✅ 成功处理 {input_filename}: 提取了 {processed_count} 条有效样本，保存至 {output_filename}"
        )

    def process(self):
        """执行完整的数据预处理流程"""
        print("开始进行第一阶段数据处理 => Processed JSONL)...")

        # 依次处理训练、验证、测试集
        self._process_file("train.json", "train.jsonl")
        self._process_file("val.json", "val.jsonl")
        self._process_file("test.json", "test.jsonl")

        print("\n🎉 第一阶段数据处理完成！")
