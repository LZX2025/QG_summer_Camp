# 导入必需的库
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch
import jieba
import json

import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super(TransformerDecoderModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, embed_size)).to(device)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        # 堆叠多个解码器层构成完整的解码器
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 定义输出层，将解码器输出转换回词汇空间
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        # 嵌入输入并添加位置编码
        src = self.embed(src) + self.positional_encoding
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
        output = self.transformer_decoder(src, src, src_mask)
        output = self.fc(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        # 生成一个上三角矩阵，用于序列生成中遮蔽未来位置的信息
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TextDataset(Dataset):
    # 初始化函数，filepath为输入文件路径
    def __init__(self, filepath):
        words = []

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # 使用jieba库进行分词，并去除每行的首尾空白字符
                words.extend(list(jieba.cut(line.strip())))

        # 将所有单词转换为一个集合来去除重复，然后再转回列表形式，形成词汇表
        self.vocab = list(set(words))
        self.vocab_size = len(self.vocab)
        self.word_to_int = {word: i for i, word in enumerate(self.vocab)}
        self.int_to_word = {i: word for i, word in enumerate(self.vocab)}

        with open('word_to_int.json', 'w', encoding='utf-8') as f:
            json.dump(self.word_to_int, f, ensure_ascii=False, indent=4)
        with open('int_to_word.json', 'w', encoding='utf-8') as f:
            json.dump(self.int_to_word, f, ensure_ascii=False, indent=4)

        # 将所有单词转换为对应的整数索引，形成数据列表
        self.data = [self.word_to_int[word] for word in words]

    # 返回数据集的长度减1，这通常是因为在机器学习中可能需要使用当前数据点预测下一个数据点
    def __len__(self):
        return len(self.data) - 1

    # 根据索引idx返回数据，这里用于返回模型训练时的输入序列和目标输出
    def __getitem__(self, idx):
        # 从数据中提取最多50个整数索引作为输入序列
        input_seq = torch.tensor(self.data[max(0, idx - 50):idx], dtype=torch.long)
        # 提取目标输出，即索引位置的单词
        target = torch.tensor(self.data[idx], dtype=torch.long)
        return input_seq, target  # 返回一个元组包含输入序列和目标输出


def custom_collate(batch):
    """自定义数据加载器的collate函数，处理可变长度序列"""
    inputs = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])


    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    return inputs_padded, targets

model_path = 't_model.pth'
data_path = "../data/DoubanConversationCorpus/test.txt"
def main():

    dataset = TextDataset(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    model = TransformerDecoderModel(vocab_size=dataset.vocab_size, embed_size=64, num_heads=2, hidden_dim=256, num_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model_path = 't_model.pth'
    num_epochs = 2
    model.train()


    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 将输入数据转置，以符合模型的期望输入维度
            inputs = inputs.t()
            optimizer.zero_grad()
            outputs = model(inputs)
            # 选择输出的最后一个元素进行损失计算
            outputs = outputs[-1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item()}')
                torch.save(model.state_dict(), model_path)


    torch.save(model, model_path)
    print('模型已保存到', model_path)
