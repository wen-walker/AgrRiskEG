import json
from tqdm import tqdm
from collections import defaultdict


class Processor(object):
    def __init__(self, config):
        self.config = config

    def _parse_single_task(self, task):
        """
        核心解析逻辑：将 Label Studio 的 Graph 结构解析为线性化的三元组字符串
        新格式：<前件|触发词|后件> <前件||后件>
        """
        # 1. 获取输入文本
        raw_text = task.get("data", {}).get("text", "")
        if not raw_text:
            return None

        # 添加任务前缀 (Prompt)
        input_text = f"抽取农业灾害三元组，格式为<前件|触发词|后件>：{raw_text}"

        # 2. 获取标注数据
        if not task.get("annotations"):
            return None
        result = task["annotations"][0]["result"]

        # --- 第一步：构建节点映射 (ID -> Entity Info) ---
        nodes = {}
        for item in result:
            if item["type"] == "labels":
                node_id = item["id"]
                # 提取文本内容和标签类型 (EVENT1, TRIG, EVENT2)
                text = item["value"]["text"]
                label = item["value"]["labels"][0]
                nodes[node_id] = {"text": text, "label": label}

        # --- 第二步：构建图连接 (Adjacency List) ---
        # adj: 正向图 (from -> to)，用于找宾语
        adj = defaultdict(list)
        # rev_adj: 反向图 (to -> from)，用于找主语
        rev_adj = defaultdict(list)

        for item in result:
            if item["type"] == "relation":
                from_id = item["from_id"]
                to_id = item["to_id"]
                adj[from_id].append(to_id)
                rev_adj[to_id].append(from_id)

        # --- 第三步：生成三元组 ---
        triples = set()  # 使用集合自动去重

        # 遍历所有节点，寻找成对关系
        for node_id, node_info in nodes.items():

            # 策略 A: 显式触发词模式 (Subject -> TRIG -> Object)
            if node_info["label"] == "TRIG":
                trigger_text = node_info["text"]
                # 谁指向了该触发词 (Subjects)
                subjects = [
                    nodes[sid]["text"] for sid in rev_adj[node_id] if sid in nodes
                ]
                # 该触发词指向了谁 (Objects)
                objects = [nodes[oid]["text"] for oid in adj[node_id] if oid in nodes]

                # 笛卡尔积组合 (处理一对多、多对一)
                for s in subjects:
                    for o in objects:
                        # 修改点：使用 <A|B|C> 格式，彻底避免逗号冲突
                        triples.add(f"<{s}|{trigger_text}|{o}>")

            # 策略 B: 隐式直连模式 (Event -> Event，无 Trigger)
            # 如果当前节点指向另一个节点，且两者都不是 TRIG
            elif node_info["label"] != "TRIG":
                targets = adj[node_id]
                for target_id in targets:
                    if target_id in nodes:
                        target_info = nodes[target_id]
                        # 确保目标也不是 Trigger (避免策略A重复)
                        if target_info["label"] != "TRIG":
                            s = node_info["text"]
                            o = target_info["text"]
                            # 修改点：中间留空，双竖线表示无触发词
                            triples.add(f"<{s}||{o}>")

        # --- 第四步：格式化输出 Target ---
        if not triples:
            target_text = "无"
        else:
            # 排序以保证训练稳定性
            sorted_triples = sorted(list(triples))
            # 修改点：使用空格分隔多个三元组，看起来像 <A|B|C> <D||E>
            target_text = " ".join(sorted_triples)

        return {"input_text": input_text, "target_text": target_text}

    def process_raw_data(self, raw_data):
        """
        批量处理原始数据列表
        """
        processed_data = []
        # Label Studio 导出通常是一个列表，每个元素是一个 Task
        for task in raw_data:
            parsed_item = self._parse_single_task(task)
            if parsed_item:
                processed_data.append(parsed_item)
        return processed_data

    def process(self):
        datasets = ["train", "val", "test"]

        for dataset_name in tqdm(datasets, desc="🚀 处理数据集", colour="cyan"):
            file_path = (
                self.config.RAW_DATA_DIR / self.config.dataset / f"{dataset_name}.json"
            )
            # 读取原始数据
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as e:
                # 兼容性处理：如果找不到 train.json 但有 example.json，且当前是处理 train
                if (
                    dataset_name == "train"
                    and (self.config.RAW_DATA_DIR / "example.json").exists()
                ):
                    print(f"ℹ️ 未找到 train.json，尝试读取 example.json...")
                    try:
                        with open(
                            self.config.RAW_DATA_DIR / "example.json",
                            "r",
                            encoding="utf-8",
                        ) as f:
                            raw_data = json.load(f)
                    except Exception as sub_e:
                        print(f"❌ 读取 example.json 也失败: {sub_e}")
                        continue
                else:
                    print(f"❌ 读取文件 {file_path} 失败: {e}")
                    continue

            # 执行转换
            t5_ready_data = self.process_raw_data(raw_data)

            # 保存结果
            save_path = (
                self.config.PROCESSED_DATA_DIR
                / self.config.dataset
                / f"{dataset_name}_processed.json"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(t5_ready_data, f, ensure_ascii=False, indent=2)

            print(
                f"✅ {dataset_name} 处理完成，保存至: {save_path} (共 {len(t5_ready_data)} 条)"
            )
