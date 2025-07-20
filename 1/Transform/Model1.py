import torch
from torch import nn

# 定义一个仅包含解码器的Transformer模型
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super(TransformerDecoderModel, self).__init__()  # 调用基类的初始化函数
        # 创建嵌入层，将词索引转换为嵌入向量
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 初始化位置编码，是一个可学习的参数
        self.positional_encoding = nn.Parameter(torch.randn(embed_size).unsqueeze(0))
        # 定义一个Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        # 堆叠多个解码器层构成完整的解码器
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 定义输出层，将解码器输出转换回词汇空间
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        # 嵌入输入并添加位置编码
        src = self.embed(src) + self.positional_encoding
        # 生成源序列的掩码，用于屏蔽未来的信息
        src_mask = self.generate_square_subsequent_mask(src.size(0))
        # 通过解码器传递源数据和掩码
        output = self.transformer_decoder(src, src, src_mask)
        # 应用线性层输出最终的预测结果
        output = self.fc(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        # 生成一个上三角矩阵，用于序列生成中遮蔽未来位置的信息
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # 将掩码的非零位置设为无穷大，零位置设为0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


