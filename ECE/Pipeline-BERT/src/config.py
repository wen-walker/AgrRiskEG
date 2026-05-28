from pathlib import Path

# 路径配置
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / "cache"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_SAVE_DIR = ROOT_DIR / "models"


# 模型配置
class Config:
    """模型配置类"""

    ROOT_DIR = ROOT_DIR
    CACHE_DIR = CACHE_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    MODEL_SAVE_DIR = MODEL_SAVE_DIR

    def __init__(self, args=None):
        self.dataset = "example"

        self.max_len = 512

        self.epochs = 3
        self.batch_size = 2

        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.warm_factor = 0.1
        self.clip_grad_norm = 1.0

        self.bert_name = "hfl/chinese-roberta-wwm-ext"
        self.use_bert_last_4_layers = True
        self.bert_learning_rate = 5e-5

        # 命令行参数覆盖配置文件
        if args:
            for k, v in args.__dict__.items():
                if v is not None:
                    self.__dict__[k] = v

        labels_org = ["EVENT1", "EVENT2", "TRIG"]
        self.save_path = MODEL_SAVE_DIR / args.save_path if args and args.save_path else MODEL_SAVE_DIR / "model.pth"
        # BIO
        labels_list = []
        labels_list.append("Other")
        for label in labels_org:
            labels_list.append("B-" + label)
            labels_list.append("I-" + label)
        self.labels = {label: i for i, label in enumerate(labels_list)}

    def __repr__(self):
        return "{}".format(self.__dict__.items())
