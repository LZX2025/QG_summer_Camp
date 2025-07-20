
import torch
import json
import jieba
from Train1 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

def load_model(model_path):
    dataset = TextDataset(data_path)
    model = TransformerDecoderModel(vocab_size=dataset.vocab_size, embed_size=64, num_heads=2, hidden_dim=256,
                                    num_layers=2).to(device)
    state = torch.load(model_path, map_location=torch.device(device),)
    model.load_state_dict(state)
    # 设置为评估模式
    model.eval()
    return model

def load_vocab(json_file):
    """从JSON文件中加载词汇表。"""
    # 读取词汇表文件
    with open(json_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def predict(model, initial_seq, max_len=50):
    # 加载数字到单词的映射
    int_to_word = load_vocab('int_to_word.json')
    # 确保模型处于评估模式
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        generated = initial_seq
        for _ in range(max_len):
            input_tensor = torch.tensor([generated], dtype=torch.long).to(device)
            output = model(input_tensor)
            predicted_idx = torch.argmax(output[:, -1], dim=-1).item()
            generated.append(predicted_idx)
            # 如果生成结束标记，则停止生成
            if predicted_idx == len(int_to_word) - 1:
                break
        return [int_to_word[str(idx)] for idx in generated]

def generate(model, input_sentence, max_len=50):
    # 使用结巴分词对输入句子进行分词
    input_words = list(jieba.cut(input_sentence.strip()))

    word_to_int = load_vocab('word_to_int.json')
    input_seq = [word_to_int.get(word, len(word_to_int) - 1) for word in input_words]

    generated_text = predict(model, input_seq, max_len)
    return "".join(generated_text)

def main():
    # 定义输入提示
    prompt = "我 要 玩 原神"
    model = load_model('t_model.pth')
    completion = generate(model, prompt)
    print(completion)

if __name__ == '__main__':
    # 主函数入口
    main()
