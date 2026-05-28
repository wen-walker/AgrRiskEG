import torch
from pathlib import Path

# 路径配置
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / "cache"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_SAVE_DIR = ROOT_DIR / "models"


# 模型配置
class Config:
    """模型配置类"""

    ROOT_DIR = ROOT_DIR
    CACHE_DIR = CACHE_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    OUTPUT_DIR = OUTPUT_DIR
    MODEL_SAVE_DIR = MODEL_SAVE_DIR

    def __init__(self, args=None):
        self.save_path = MODEL_SAVE_DIR

        self.max_length = 256
        self.max_span_len = 30
        self.batch_size = 4

        self.bert_name = "hfl/chinese-roberta-wwm-ext"
        self.use_bert_last_4_layers = False
        self.bert_learning_rate = 5e-5
        self.hidden_size = 768
        self.num_queries_exp = 10
        self.num_queries_imp = 4

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 35
        self.learning_rate = 3e-5
        self.weight_decay = 1e-4
        self.warm_factor = 0.1
        self.clip_grad_norm = 1.0

        self.weight_cls = 2.0
        self.weight_span = 1.0  # Span 匹配权重
        self.weight_len = 0.2  # 边界长度惩罚权重
        self.weight_aux = 0.5
        self.weight_ispd = 0.5

        # 命令行参数覆盖配置文件
        if args:
            for k, v in args.__dict__.items():
                if v is not None:
                    self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
