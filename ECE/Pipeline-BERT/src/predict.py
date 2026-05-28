from model import BIOBertNer, BIOBertCRFNer
import torch
from transformers import AutoTokenizer
from config import Config
import argparse

def predict_batch(input_ids, attention_mask, token_type_ids, model, config):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids) # [batch_size, seq_len, num_labels]
    logits = outputs.logits
    if config.model_name == 'BIOBertNer':
        logits = torch.argmax(logits, dim=-1) # [batch_size, seq_len]
    # print(logits)
    return logits.tolist()

def bio_decode(text_chs, predict, config):
    id2label = {v: k for k, v in config.labels.items()}
    out = []
    i = 0
    while i < len(predict):
        tag = id2label[predict[i]]
        if tag.startswith('B-'):
            ent_type = tag[2:]  # 去掉前缀 B-，拿到真正的实体类型，例如 'B-Time' 变成 'Time'。
            start = i
            i += 1
            while i < len(predict) and id2label[predict[i]] == f'I-{ent_type}':
                i += 1
            surface = ''.join(text_chs[start:i])  # 把这一段 token 列表拼成字符串，得到实体的表面写法，例如 '9名伤员'。
            out.append(f'【{surface}, {ent_type}】')
        else:
            # 非实体直接原样输出
            out.append(text_chs[i])
            i += 1
    return ''.join(out)

def predict_text(text, model, tokenizer, config):
    text_chs = list(text)
    inputs = tokenizer(
        text_chs,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(config.device) # [1, seq_len]
    attention_mask = inputs['attention_mask'].to(config.device) # [1, seq_len]
    token_type_ids = inputs['token_type_ids'].to(config.device) # [1, seq_len]
    predict = predict_batch(input_ids, attention_mask, token_type_ids, model, config)[0] # [seq_len]
    # 对齐预测结果到原始文本
    perdict = predict[1:-1]  # 去掉 [CLS] 和 [SEP]
    result = bio_decode(text_chs, perdict, config)
    return result


def run_predict(config):
    # 1.加载模型
    print("🔄 模型加载中...")
    if config.model_name == 'BIOBertNer':  
        model = BIOBertNer(config)
    elif config.model_name == 'BIOBertCRFNer':  
        model = BIOBertCRFNer(config)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.to(config.device)
    print("📥 模型加载成功！")
    # 2.加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir=str(config.CACHE_DIR))
    # 3.命令行交互循环
    print("🤖 欢迎使用BIO命名实体识别系统！")
    print("输入文本，按回车进行实体识别；输入（输入q或quit退出）退出系统。")
    while True:
        text = input("User >>").strip()
        if text.lower() in ['q', 'quit']:
            print("👋 再见。")
            break
        if len(text) == 0:
            print("输入不能为空，请重新输入。")
            continue
        result = predict_text(text, model, tokenizer, config)
        print("识别结果：", result)

if __name__ == '__main__':
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BIOBertNer', help='模型名称')
    parser.add_argument('--save_path', type=str, default='model_en.pth', help='模型保存路径')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=device, help='训练设备')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--bert_name', type=str, default=None, help='预训练模型名称')
    args = parser.parse_args()

    # 2. 加载配置
    config = Config(args)
    run_predict(config)
