from pathlib import Path

# 路径配置
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / "cache"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_SAVE_DIR = ROOT_DIR / "models"
CORPUS_DIR = ROOT_DIR / "data" / "corpus"
RESULTS_DIR = ROOT_DIR / "data" / "results"

# 模型配置
class Config:
    """模型配置类"""

    ROOT_DIR = ROOT_DIR
    CACHE_DIR = CACHE_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    MODEL_SAVE_DIR = MODEL_SAVE_DIR
    CORPUS_DIR = CORPUS_DIR
    RESULTS_DIR = RESULTS_DIR

    def __init__(self, args=None):
        self.dataset = "example"
        self.t5_name = "Langboat/mengzi-t5-base"
        self.save_path = MODEL_SAVE_DIR
        self.corpus_path = CORPUS_DIR / "example.txt"
        self.results_path = RESULTS_DIR / "results.txt"
        self.fp16 = False
        self.max_source_len = 256
        self.max_target_len = 128

        self.epochs = 3
        self.batch_size = 1

        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.warm_factor = 0.1
        self.clip_grad_norm = 1.0

        # 命令行参数覆盖配置文件
        if args:
            for k, v in args.__dict__.items():
                if v is not None:
                    self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
