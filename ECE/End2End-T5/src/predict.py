import torch
import re
from transformers import T5ForConditionalGeneration, AutoTokenizer
from config import Config
import logging

# 禁止 transformers 的一些啰嗦警告
logging.getLogger("transformers").setLevel(logging.ERROR)


class InferenceEngine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型加载路径 (默认加载最佳模型)
        self.model_path = config.MODEL_SAVE_DIR / "best_model"

        print(f"🔄 正在加载模型权重: {self.model_path} ...")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到模型文件，请先运行 train.py: {self.model_path}"
            )

        # 建议使用 AutoTokenizer，兼容性更好
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model.eval()
        print(f"✅ 模型加载完成! 设备: {self.device}")

    def parse_output(self, text):
        """
        将模型生成的字符串解析为结构化列表 (适配 <E1|T|E2> 格式)
        输入: "<越冬期干旱|造成|减产> <高温||玉米绝收>"
        输出: [{'前件': '越冬期干旱', '触发词': '造成', '后件': '减产'}, ...]
        """
        triples = []
        text = text.strip()

        if text == "无" or not text:
            return triples

        # 1. 使用正则表达式提取所有尖括号内的内容
        # pattern: 匹配 < 和 > 之间的所有非 > 字符
        pattern = r"<([^>]+)>"
        matches = re.findall(pattern, text)

        for content in matches:
            # content 现在是 "越冬期干旱|造成|减产"
            # 2. 按竖线分割
            parts = content.split("|")

            # 确保分割出3部分 (前件|触发词|后件)
            # 我们的格式严格保证有两个竖线
            if len(parts) == 3:
                s = parts[0].strip()
                p = parts[1].strip()
                o = parts[2].strip()

                # 只有当三元组的前件或后件不为空时才添加
                if s or o:
                    triples.append(
                        {
                            "前件": s,
                            "触发词": (
                                p if p else "N/A"
                            ),  # 如果触发词为空字符串，显示 N/A
                            "后件": o,
                        }
                    )
        return triples

    def predict(self, sentence):
        """
        单句预测
        """
        # 1. 数据预处理：加上训练时使用的 Prompt 前缀
        # 注意：必须与 train.py 中 data_process 的前缀保持一致
        input_text = f"农业灾害事件三元组抽取：{sentence}"

        # 2. 编码
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config.max_source_len,
            truncation=True,
        ).input_ids.to(self.device)

        # 3. 生成 (Beam Search)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=self.config.max_target_len,
                num_beams=4,  # 使用集束搜索
                early_stopping=True,
                repetition_penalty=1.2,  # 新增：惩罚重复生成的词，防止复读机
            )

        # 4. 解码
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 5. 解析结构
        structured_data = self.parse_output(decoded_text)

        return decoded_text, structured_data


def main():
    # 初始化配置和引擎
    config = Config()

    try:
        engine = InferenceEngine(config)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    print("\n" + "=" * 50)
    print("🌾 农业气象灾害事件抽取交互系统 🌾")
    print("输入句子后回车进行抽取，输入 'q' 或 'quit' 退出。")
    print("=" * 50 + "\n")

    while True:
        # 获取用户输入
        try:
            raw_text = input("\n👉 请输入句子: ").strip()
        except KeyboardInterrupt:
            print("\n退出...")
            break

        # 退出条件
        if raw_text.lower() in ["q", "quit", "exit"]:
            print("👋 再见！")
            break

        if not raw_text:
            continue

        # 执行预测
        raw_output, structured_triples = engine.predict(raw_text)

        # 格式化输出
        print("-" * 30)
        print(f"🤖 模型原始输出: {raw_output}")
        print("-" * 30)

        if not structured_triples:
            print("🤷 未发现有效的三元组信息。")
        else:
            print(f"✅ 抽取结果 ({len(structured_triples)}组):")
            for i, item in enumerate(structured_triples, 1):
                # 美化打印
                t_str = item["触发词"]
                # 如果触发词是 N/A (即空)，显示为隐式关系
                relation_display = (
                    f" --[{t_str}]--> " if t_str != "N/A" else " --(直接影响)--> "
                )

                print(f"   {i}. {item['前件']}{relation_display}{item['后件']}")

        print("=" * 50)


if __name__ == "__main__":
    main()
