import json
import os
from pathlib import Path
from tqdm import tqdm
import random  # 1. 引入 random


class EntityProcessor:
    def __init__(self, config):
        """
        初始化处理器
        :param config: 配置类实例
        """
        self.config = config
        # 支持处理 train、test、val 三种数据集
        self.datasets = ["train", "test", "val"]

        # 确保输出目录存在
        if not (self.config.PROCESSED_DATA_DIR / config.dataset).exists():
            (self.config.PROCESSED_DATA_DIR / config.dataset).mkdir(parents=True)

    def _load_data(self, raw_path):
        """读取原始 JSON 文件"""
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        with open(raw_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _convert_label_to_id(self, label_str):
        """将 BIO 字符串标签转换为 ID，如果不存在则返回 Other 的 ID"""
        return self.config.labels.get(label_str, self.config.labels["Other"])

    def _process_single_sample(self, item):
        """处理单个样本"""
        # 1. 获取原始文本并按字符切分
        text_str = item["data"]["text"]
        # 将字符串转换为字符列表
        char_list = list(text_str)

        # 2. 初始化标签列表，默认为 "Other"
        # 使用临时列表存储字符串标签 (e.g., "B-EVENT1", "O")
        bio_labels = ["Other"] * len(char_list)

        # 3. 解析标注信息
        # 检查是否有标注数据
        if item.get("annotations"):
            # Label Studio 通常把结果放在 annotations[0]['result'] 中
            annotation_results = item["annotations"][0].get("result", [])

            for res in annotation_results:
                # 我们只处理实体识别 (labels 类型)，跳过关系抽取 (relation 类型)
                if res.get("type") != "labels":
                    continue

                value = res.get("value", {})
                start = value.get("start")
                end = value.get("end")
                # Label Studio 的 labels 是一个列表，通常 NER 只有一个标签
                label_name = value.get("labels", [])[0] if value.get("labels") else None

                if start is not None and end is not None and label_name:
                    # 确保索引不越界
                    start = max(0, start)
                    end = min(len(char_list), end)

                    # 标记 B-Tag (开始位置)
                    bio_labels[start] = f"B-{label_name}"

                    # 标记 I-Tag (中间位置)
                    # Label Studio 的 end 是 exclusive 的 (不包含 end 本身)，符合 Python切片习惯
                    for i in range(start + 1, end):
                        bio_labels[i] = f"I-{label_name}"

        # 4. 将字符串标签转换为 ID
        label_ids = [self._convert_label_to_id(l) for l in bio_labels]

        # 5. 构建样本
        sample = {"text": char_list, "label": label_ids}
        return sample

    def process(self):
        """执行数据处理流程，处理 train、test、val 三个数据集"""
        for dataset_type in self.datasets:
            raw_path = (
                self.config.RAW_DATA_DIR / self.config.dataset / f"{dataset_type}.json"
            )
            save_path = (
                self.config.PROCESSED_DATA_DIR
                / self.config.dataset
                / f"{dataset_type}_processed.json"
            )

            # 如果原始文件不存在，则跳过
            if not raw_path.exists():
                print(f"Warning: Raw data file does not exist, skipping: {raw_path}")
                continue

            # 加载数据
            data = self._load_data(raw_path)
            processed_data = []

            # 使用 tqdm 显示进度
            for item in tqdm(data, desc=f"🚀 构造 {dataset_type.title()} 实体标签", colour='cyan'):
                sample = self._process_single_sample(item)
                processed_data.append(sample)

            # 保存处理后的文件
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)


class RelationProcessor:
    def __init__(self, config):
        self.config = config
        # 支持处理 train、test、val 三种数据集
        self.datasets = ["train", "test", "val"]

        # 确保输出目录存在
        if not self.config.PROCESSED_DATA_DIR.exists():
            self.config.PROCESSED_DATA_DIR.mkdir(parents=True)

    def _process_single_item(self, item, dataset_type):  # 2. 增加 dataset_type 参数
        """
        处理单个项目，提取关系信息
        """
        text = item["data"]["text"]

        # 如果没有标注数据，返回空列表
        if not item.get("annotations"):
            return [], 0

        results = item["annotations"][0].get("result", [])

        # --- 步骤 1: 扫描所有节点 (Entities) ---
        id_to_info = {}
        id_to_type = {}

        for res in results:
            if res["type"] == "labels":
                label_val = res["value"]["labels"][0]
                node_id = res["id"]
                id_to_info[node_id] = {
                    "id": node_id,
                    "start": res["value"]["start"],
                    "end": res["value"]["end"],
                    "text": res["value"]["text"],
                    "label": label_val,
                }
                if "TRIG" in label_val:
                    id_to_type[node_id] = "TRIG"
                elif "EVENT" in label_val:
                    id_to_type[node_id] = "EVENT"

        # --- 步骤 2: 构建邻接表 ---
        adj = {}
        for res in results:
            if res["type"] == "relation":
                src = res["from_id"]
                dst = res["to_id"]
                if src not in adj:
                    adj[src] = []
                adj[src].append(dst)

        # --- 步骤 3: 逻辑推导真实关系 ---
        # 改为 dict: (src_id, dst_id) -> has_trig (bool)
        # has_trig=True  → 显式关系（路径经过 TRIG 节点）
        # has_trig=False → 隐式关系（EVENT 直接指向 EVENT）
        true_relations = {}
        for src_id, targets in adj.items():
            if id_to_type.get(src_id) != "EVENT":
                continue
            for target_id in targets:
                target_type = id_to_type.get(target_id)
                if target_type == "EVENT":
                    # 直接连接：隐式
                    true_relations[(src_id, target_id)] = False
                elif target_type == "TRIG":
                    if target_id in adj:
                        for next_hop_id in adj[target_id]:
                            if id_to_type.get(next_hop_id) == "EVENT":
                                # 经过触发词：显式
                                true_relations[(src_id, next_hop_id)] = True

        # --- 步骤 4: 生成训练样本 (Pair Generation) ---
        events = [info for nid, info in id_to_info.items() if "EVENT" in info["label"]]
        re_samples = []

        for i in range(len(events)):
            for j in range(len(events)):
                if i == j:
                    continue

                e1 = events[i]
                e2 = events[j]

                # 判断是否存在关系，以及是否经过触发词
                pair_key = (e1["id"], e2["id"])
                is_related = pair_key in true_relations
                has_trig   = true_relations.get(pair_key, False)
                label = 1 if is_related else 0

                # =======================================================
                # 【关键修改】方案一：负采样逻辑 (Negative Sampling)
                # =======================================================
                # 仅对 训练集(train) 的 负样本(label=0) 进行采样
                if dataset_type == "train" and label == 0:
                    if random.random() > 0.25:
                        continue
                # =======================================================

                # 构建样本
                sample = {
                    "text": text,
                    "e1_text": e1["text"],
                    "e2_text": e2["text"],
                    "e1_start": e1["start"],
                    "e1_end": e1["end"],
                    "e2_start": e2["start"],
                    "e2_end": e2["end"],
                    "label": label,
                    "has_trig": has_trig,
                }
                re_samples.append(sample)

        return re_samples, len(true_relations)

    def process(self):
        """
        主处理函数
        """
        for dataset_type in self.datasets:
            # 路径拼接
            raw_path = (
                self.config.RAW_DATA_DIR / self.config.dataset / f"{dataset_type}.json"
            )
            save_path = (
                self.config.PROCESSED_DATA_DIR
                / self.config.dataset
                / f"{dataset_type}_re.json"
            )

            if not raw_path.exists():
                print(f"Warning: Raw data file does not exist, skipping: {raw_path}")
                continue

            print(f"Processing {dataset_type} relation data from {raw_path}...")

            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            re_samples = []
            total_true_relations = 0

            for item in tqdm(
                data, desc=f"🚀 构造 {dataset_type.upper()} 关系标签", colour="cyan"
            ):
                # 3. 调用时传入 dataset_type
                samples, relations_count = self._process_single_item(item, dataset_type)
                re_samples.extend(samples)
                total_true_relations += relations_count

            # --- 结果统计与保存 ---
            print("=" * 30)
            print(f"{dataset_type} 数据处理完成:")
            print(f"  - 原始逻辑关系数 (Positive Pairs): {total_true_relations}")
            print(f"  - 生成总样本数 (Total Samples): {len(re_samples)}")

            pos_samples = sum(s["label"] for s in re_samples)
            neg_samples = len(re_samples) - pos_samples

            # 打印比例，方便检查采样效果
            ratio = f"1:{neg_samples/pos_samples:.1f}" if pos_samples > 0 else "N/A"
            print(f"  - 正样本 (Label=1): {pos_samples}")
            print(f"  - 负样本 (Label=0): {neg_samples}")
            print(f"  - 正负比例: {ratio}")
            print("=" * 30)

            print(f"Saving {dataset_type} RE data to {save_path}...")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(re_samples, f, ensure_ascii=False, indent=2)

            print(f"Done processing {dataset_type} relation data.\n")
