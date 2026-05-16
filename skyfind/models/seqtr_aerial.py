# 文件路径: models/seqtr_aerial.py
import torch
import torch.nn as nn
import math

# 导入设计的两个编码器
from models.encoders import DarkNet53, GRUTextEncoder

class PositionalEncoding(nn.Module):
    """
    位置编码：Transformer 本身没有顺序概念，必须通过正弦/余弦信号告诉它单词或图像块的位置。
    每个token ID位置都有一个位置向量（维度与embedding vector的维度相同，因为后面还要两者相加）
    """
    def __init__(self, d_model: int = 256, max_len: int = 2000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)#所有偶数维度使用sin函数
        pe[0, :, 1::2] = torch.cos(position * div_term)#所有奇数维度使用cos函数
        self.register_buffer('pe', pe)#非参数，不参与梯度更新，但会随模型一起保存

    def forward(self, x):
        # x shape: [B, Seq_len, d_model]
        #使用广播机制将位置编码加到embedding vector上面
        return x + self.pe[:, :x.size(1)]

class AerialRECSeqTR(nn.Module):
    """
    基于 SeqTR 架构，并融入 AerialREC 双阶段 (PTR + PTBB) 机制的完整模型。
    对应论文中的 AerialREC†。
    """
    def __init__(self, hidden_dim: int = 256, num_bins: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 初始化两个基石编码器
        self.visual_encoder = DarkNet53(hidden_dim=hidden_dim)
        # 词表 30522 是 BERT 默认大小
        self.text_encoder = GRUTextEncoder(vocab_size=30522, embed_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)
        
        # 3. Transformer 编码器 (用于图文特征融合 Fusion)
        # 论文机理：图像特征和文本特征拼在一起，让它们利用 Self-Attention 互相认识。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=1024, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 4. Transformer 解码器 (用于自回归预测带 <SEP> 的坐标序列)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=1024, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # 解码器的坐标 Token 词嵌入
        # 词表大小 = num_bins(1000) + BOS + EOS + SEP + PAD = 1004
        self.vocab_size = num_bins + 4
        self.tgt_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        
        # 5. 最后的分类头：将 256 维的隐藏状态预测为 1004 个类别中的一个！！！为什么是256维？
        self.head = nn.Linear(hidden_dim, self.vocab_size)

    #！！！
    def generate_square_subsequent_mask(self, sz: int):
        """
        生成下三角掩码矩阵，确保解码器在预测第 t 个词时，看不到 t 以后的词 (Causal Mask)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    #！！！
    def forward(self, images, text_ids, text_masks, tgt_seq):
        """
        训练时的前向传播 (Teacher Forcing 模式)
        
        :param images: [B, 3, 640, 640]
        :param text_ids: [B, text_len]
        :param text_masks:[B, text_len] (1表示有效，0表示Pad)
        :param tgt_seq: [B, tgt_len] (我们的带 SEP 的序列，例如长为 11)
        :return: 坐标单词的预测概率分布 [B, tgt_len-1, vocab_size]
        """
        B = images.size(0)
        
        # --- 步骤 1: 提取独立特征 ---
        # vis_feat:[B, 400, 256]
        vis_feat = self.visual_encoder(images)
        # txt_feat: [B, text_len, 256]
        txt_feat = self.text_encoder(text_ids, attention_mask=text_masks)
        
        # --- 步骤 2: 多模态融合 (Multi-modal Fusion) ---
        # 沿着长度维度拼起来 -> [B, 400 + text_len, 256]
        memory_input = torch.cat([vis_feat, txt_feat], dim=1)
        # 加上位置编码，否则 Transformer 成了瞎子，不知道谁在左谁在右
        memory_input = self.pos_encoder(memory_input)

        #===================================================
        '''防污染机制：构建全局填充掩码
        Pytorch的src_key_padding_mask是一个布尔矩阵，形状为 [batch_size, src_len]，
        其中 True 表示对应位置是填充（padding,必须忽略），False 表示对应位置是有效的输入（参与计算注意力）。
    
        '''
        #图像部分的掩码（400个图像块）全是 “False”，因为图像特征都是有效的
        num_vis = vis_feat.size(1) # 400
        vis_pad_mask = torch.zeros((B, num_vis),dtype=torch.bool, device=images.device)#[B,400]
        #文本部分的掩码，根据 text_masks 生成，1表示有效，0表示Pad，我们需要把它转换为布尔类型，并且 True 表示填充（padding）
        # text_masks 原本是 1(有效) 和 0(PAD)。我们取反，把 0 变成 True，1 变成 False。
        txt_pad_mask = (text_masks == 0).bool() #[B, text_len]
        #拼接图像掩码和文本掩码，得到完整的长度为 400 + text_len 的掩码矩阵
        memory_key_padding_mask = torch.cat([vis_pad_mask, txt_pad_mask], dim=1)
        # 传入 Transformer Encoder
        # 现在，图像和真实文本绝不会去和 PAD 交互了
        memory = self.transformer_encoder(
            memory_input, 
            src_key_padding_mask=memory_key_padding_mask
        )
        #===================================================
        
        # --- 步骤 3: 序列解码 (Sequence Decoding) ---
        # 训练时，输入给解码器的是目标序列的 "前 N-1 个词"
        # 比如序列是 [BOS, x1, y1, SEP, x2, EOS]，解码器输入为 [BOS, x1, y1, SEP, x2]
        decoder_input = tgt_seq[:, :-1]
        tgt_embed = self.tgt_embedding(decoder_input)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # 防止偷看未来的 Mask
        tgt_seq_len = decoder_input.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(images.device)
        
        # 解码器结合了刚刚融合好的图文特征 (memory) 和当前的坐标序列
        dec_out = self.transformer_decoder(
            tgt=tgt_embed, 
            memory=memory, 
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask # 让解码器交叉注意力也不去看那些 PAD 位置的特征
        )
        
        # --- 步骤 4: 预测 (Prediction) ---
        # 将输出的隐藏特征映射到词表大小的概率空间
        logits = self.head(dec_out) # -> [B, tgt_len-1, 1004]
        
        return logits

    @torch.no_grad() # 推理时绝对不能计算梯度，极其节省显存！
    def inference(self, images, text_ids, text_masks, max_len=11):
        """
        测试/验证时的前向推理 (Autoregressive Decoding)
        模型从 <BOS> 开始，一步一步自己生成后续的坐标 token。
        
        :param max_len: 生成的最大长度。我们的序列是11个 (BOS + 4个PTR + SEP + 4个PTBB + EOS)
        :return: 预测的离散坐标序列 [B, max_len]
        """
        device = images.device
        B = images.size(0)
        
        # --- 步骤 1: 提取独立特征并融合 (和训练完全一样) ---
        vis_feat = self.visual_encoder(images)
        txt_feat = self.text_encoder(text_ids, attention_mask=text_masks)
        
        memory_input = torch.cat([vis_feat, txt_feat], dim=1)
        memory_input = self.pos_encoder(memory_input)
        memory = self.transformer_encoder(memory_input) # f_m: 完美的图文融合上下文
        
        # --- 步骤 2: 自回归解码循环 (核心差异) ---
        # 1000 是 num_bins, 所以 BOS_TOKEN 默认是 1000
        BOS_TOKEN = self.vocab_size - 4 
        
        # 初始化生成的序列：现在只有一列 <BOS>，形状为 [B, 1]
        generated_seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        
        # 循环 max_len - 1 次 (因为已经有 BOS 了，还需要生成 10 个词)
        for step in range(max_len - 1):
            # 将目前已生成的序列转为词嵌入并加入位置编码
            tgt_embed = self.tgt_embedding(generated_seq)
            tgt_embed = self.pos_encoder(tgt_embed)
            
            # 生成因果掩码 (防止模型报错，虽然推理时它只能看到自己已经生成的)
            tgt_mask = self.generate_square_subsequent_mask(generated_seq.size(1)).to(device)
            
            # 解码器出马
            dec_out = self.transformer_decoder(
                tgt=tgt_embed, 
                memory=memory, 
                tgt_mask=tgt_mask
            )
            
            # 经过分类头映射为概率 logits
            logits = self.head(dec_out) # 形状: [B, 当前长度, 1004]
            
            # 我们只关心它最新“吐”出来的那个词（也就是最后一个位置的概率分布）
            next_token_logits = logits[:, -1, :] # 形状: [B, 1004]
            
            # 使用 argmax 选择概率最大的那个类作为预测结果 (贪心解码 Greedy Decoding)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True) # 形状: [B, 1]
            
            # 将新预测的词拼接回原来的序列中
            generated_seq = torch.cat([generated_seq, next_token], dim=1)
            
            # (优化点) 如果想提速，可以在这里加个判断：如果所有 batch 都输出了 EOS，就可以提前 break。
            # 为了严谨复现 AerialREC 强制长度的逻辑，我们这里循环跑满 11 步。
            
        return generated_seq
# ==========================================
# 交互验证环节：跑通模型
# ==========================================
if __name__ == "__main__":
    B = 2 # 保证所有输入的 Batch Size 一致！
    dummy_imgs = torch.randn(B, 3, 640, 640)
    dummy_txt_ids = torch.randint(0, 30522, (B, 40))
    dummy_txt_masks = torch.ones(B, 40)
    
    # 模拟我们的 AerialREC 序列 (长度为11)
    #[BOS, ptr1, ptr2, ptr3, ptr4, SEP, ptbb1, ptbb2, ptbb3, ptbb4, EOS]
    dummy_tgt_seq = torch.tensor([[1000, 311, 400, 500, 600, 1002, 350, 450, 480, 580, 1001],[1000, 100, 200, 300, 400, 1002, 150, 250, 280, 380, 1001]
    ])
    
    print(" 初始化 AerialREC†(SeqTR) 完整模型...")
    model = AerialRECSeqTR(hidden_dim=256, num_bins=1000)
    
    print(" 开始多模态前向传播计算...")
    logits = model(dummy_imgs, dummy_txt_ids, dummy_txt_masks, dummy_tgt_seq)
    
    print("\n" + "="*50)
    print(" 模型整体架构 (Fusion & Decoder) 跑通了！")
    print(f"最终输出 Logits 形状: {logits.shape}")
    print("物理意义解释：")
    print(f"Batch Size = {logits.size(0)}")
    print(f"预测序列长度 = {logits.size(1)} (因为去掉了最后一个词作为输入，11 - 1 = 10)")
    print(f"词表概率分布维度 = {logits.size(2)} (1000个坐标bins + 4个特殊Token)")
    print("="*50)