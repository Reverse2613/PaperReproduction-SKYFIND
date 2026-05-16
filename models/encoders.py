# 文件路径: models/encoders.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# ==========================================================
# 1. 严格复现：DarkNet-53 视觉特征提取器
# ==========================================================
class DarknetBlock(nn.Module):
    """
    DarkNet-53 的基础残差块：1x1 卷积降维 + 3x3 卷积提取特征 + 残差连接
    物理意义：在保持计算效率的同时，提取丰富的空间和局部特征（为目标检测量身定制）
    """
    def __init__(self, in_channels):
        super().__init__()
        hidden_channels = in_channels // 2  #向下取整
        # 1x1 卷积压缩通道，特征图尺寸未改变
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # 3x3 卷积提取特征并恢复通道，特征图尺寸未改变
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual # 残差连接

class DarkNet53(nn.Module):
    """
    严格按照 YOLOv3 骨干网络设计的 DarkNet-53
    层数分布：1, 2, 8, 8, 4 个残差块
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # 初始层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # 封装下采样与残差块组合的辅助函数
        #！！！
        def make_layer(in_c, out_c, num_blocks):
            layers =[
                # 步长为2的 3x3 卷积进行下采样（尺寸减半）
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1, inplace=True)
            ]
            for _ in range(num_blocks):
                layers.append(DarknetBlock(out_c))
            return nn.Sequential(*layers)

        # 依次构建 DarkNet-53 的 5 个阶段 (下采样 5 次，总计降采样 2^5 = 32倍)
        self.layer1 = make_layer(32, 64, 1)
        self.layer2 = make_layer(64, 128, 2)
        self.layer3 = make_layer(128, 256, 8)
        self.layer4 = make_layer(256, 512, 8)
        self.layer5 = make_layer(512, 1024, 4) # 最终输出通道数为 1024

        # 降维投影层：将 1024 维统一压缩到多模态交互所需的 hidden_dim (256维)
        self.input_proj = nn.Conv2d(1024, hidden_dim, kernel_size=1)

    def forward(self, images: torch.Tensor):
        # images 形状 [B, 3, 640, 640]
        x = self.conv1(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x) # ->[B, 1024, 20, 20]

        # 降维到 256
        features = self.input_proj(x) # -> [B, 256, 20, 20]
        
        # 展平为序列: [B, 256, 400]
        features = features.flatten(2)#！！！flatten(2) 从第2维开始展平，即把20x20的空间维度展成400
        # 转置以适应后续 Transformer: [B, 400, 256]
        features = features.permute(0, 2, 1)
        return features

# ==========================================================
# 2. 严格复现：GRU 文本特征提取器
# ==========================================================
class GRUTextEncoder(nn.Module):
    """
    基于 GRU 的轻量级文本编码器
    物理意义：按时间步顺序读取词向量，最终提取序列特征。
    """
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 256, hidden_dim: int = 256):
        """
        :param vocab_size: 词表大小 (我们之前在 dataset.py 用了 bert_tokenizer，词表约30522)
        :param embed_dim: 词向量维度
        :param hidden_dim: GRU的隐藏层维度
        """
        super().__init__()
        # 将离散的 token ID 映射为连续的词向量
        """
        输入: tensor([101, 202, 305, 0, 0])  # 一个句子序列分词后的token IDs
                │
                ▼
        输出: tensor([[ 0.1, 0.3, ..., -0.2],   # 每个ID对应一个embed_dim维向量
              [ 0.5, -0.1, ...,  0.8],
              [-0.2,  0.7, ...,  0.1],
              [ 0.0,  0.0, ...,  0.0],   # padding为全0
              [ 0.0,  0.0, ...,  0.0]])
        """
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 双向 GRU (Bidirectional GRU)，能同时获取前后文信息
        # 输出维度是 hidden_dim // 2，双向拼接后刚好等于 hidden_dim
        self.gru = nn.GRU(
            input_size=embed_dim,  # 输入维度 = 词向量维度，初始化是256
            hidden_size=hidden_dim // 2, # 隐藏层维度（双向各一半），单向 GRU 的隐藏层维度，两个方向拼接后为256维
            num_layers=1,   #只有一层GRU
            batch_first=True, # 输入格式: [batch, seq, feature]
            bidirectional=True #双向，前向与后向
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor ):
        """
        :param input_ids: [B, seq_len]
        """
        #引入了文本掩码，改进了GRU的双向污染
        # 1. 获取词向量[B, seq_len, embed_dim]
        embedded = self.embedding(input_ids)
        
        # --- 严谨的防污染机制：动态打包序列 ---
        # 计算每个 batch 中真实句子的长度 (attention_mask 里 1 的个数)
        # 必须转到 CPU，因为 pack_padded_sequence 要求 lengths 在 CPU 上
        lengths = attention_mask.sum(dim=1).cpu()

        # 压紧序列：把 PAD 剔除，只保留真实的词
        # enforce_sorted=False 允许 batch 里的长度是乱序的
        packed_input = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        # 2. 喂入 GRU
        #此时GRU只会在真实的词上进行计算，完全不受 PAD 的干扰，确保了训练的纯净性和效率
        packed_output, _ = self.gru(packed_input)

        # gru_out 包含了每个时间步(单词)的隐藏层状态,就是GRU的记忆(当下时刻的)
        #返回了每个时间步GRU的隐层状态，方便拿来后续的注意力机制计算与直接分类
        #gru_out表示每个词的位置特征表示，转换为每个词位置对应的 hidden_dim 维特征向量序列，保留了完整的上下文信息
        # 形状:[B, seq_len, hidden_dim]
        #输出一个形状为 [Batch_Size, seq_len, hidden_dim] 的张量，里面每一个 hidden_dim 维的向量，不仅包含了单词本身的含义，还包含了整个句子的上下文语境
        # 松开序列：把剔除的 PAD 再用 0 填回来，恢复 [B, seq_len, hidden_dim] 形状
        gru_out, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=embedded.size(1)
        )
        
        return gru_out

# ==========================================
# 验证代码
# ==========================================
if __name__ == "__main__":
    dummy_images = torch.randn(5, 3, 640, 640)
    #随机整数[0,10000）
    dummy_input_ids = torch.randint(0, 10000, (5, 70)) #torch.randint(low最小值, high最大值, size输出张量形状)
    attention_mask = (dummy_input_ids != 0).long() # 1表示有效，0表示Pad

    print(" 初始化 DarkNet-53 视觉编码器和 GRU 文本编码器...")
    vis_enc = DarkNet53(hidden_dim=256)
    txt_enc = GRUTextEncoder(vocab_size=30522, embed_dim=256, hidden_dim=256)
    
    vis_features = vis_enc(dummy_images)
    txt_features = txt_enc(dummy_input_ids,attention_mask)
    
    print("\n" + "="*50)
    print(" 原汁原味复现验证：")
    print(f"输出text_mask: {attention_mask.shape}")
    print(f"输出【DarkNet-53视觉特征】形状: {vis_features.shape}")
    print(f"输出【GRU文本特征】形状: {txt_features.shape}")
    print("="*50)
    print(" 结论：多模态特征维度对齐完成，完全遵照 Table 4 的 SeqTR 基线设计！")