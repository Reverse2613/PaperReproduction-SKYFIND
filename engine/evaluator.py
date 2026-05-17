# 文件路径: engine/evaluator.py
import torch
import numpy as np

class AerialRECEvaluator:
    """
    负责自回归推理、坐标逆映射以及 IoU 评测的模块。
    绝无黑盒：每一个坐标的解析都做了严密的边界防护。
    """
    def __init__(self, num_bins: int = 1000):
        self.num_bins = num_bins
        # 特殊 Token 的定义 (必须和 transforms.py 保持绝对一致)
        self.BOS_TOKEN = num_bins
        self.EOS_TOKEN = num_bins + 1
        self.SEP_TOKEN = num_bins + 2
        
    def box_iou(self, box1: list, box2: list) -> float:
        """
        几何计算：计算两个 Box 的交并比 (IoU)
        :param box1: [x1, y1, x2, y2]
        :param box2: [x1, y1, x2, y2]
        :return: iou (float, 0~1)
        """
        # 1. 计算交集坐标 (两个框的重叠区域也是个矩形)
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # 如果没有交集，右边框反而比左边框小
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # 计算交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # 2. 计算各自的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 3. 计算 IoU
        union_area = box1_area + box2_area - intersection_area
        
        # 防护：除数不能为0
        if union_area <= 0:
            return 0.0
            
        return intersection_area / union_area

    def detokenize_box(self, tokens: list, img_w: float, img_h: float) -> list:
        """
        逆向离散化：将 4 个 Token ID 转回浮点数物理坐标。
        """
        # 如果模型抽风，给的不是 4 个数字，或者给了特殊符号，直接返回一个[0,0,0,0]的废框
        if len(tokens) != 4 or any(t >= self.num_bins for t in tokens):
            return [0.0, 0.0, 0.0, 0.0]
            
        # 核心逆映射公式
        x1 = (tokens[0] / (self.num_bins - 1)) * img_w
        y1 = (tokens[1] / (self.num_bins - 1)) * img_h
        x2 = (tokens[2] / (self.num_bins - 1)) * img_w
        y2 = (tokens[3] / (self.num_bins - 1)) * img_h
        
        # 确保 x1 < x2 且 y1 < y2，修正模型可能产生的逻辑反转
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def generate_sequence(self, model, images, text_ids, text_masks, max_len: int = 11):
        """
        最核心的逻辑：自回归推理生成 (贪心策略)
        脱离教师强迫（Teacher Forcing），一个词一个词蹦出来。
        """
        B = images.size(0)
        device = images.device
        
        # 1. 独立运行 Encoder，获取 Memory 档案库（算一次就够了，不用在循环里重复算！重复算浪费时间）
        vis_feat = model.visual_encoder(images)
        txt_feat = model.text_encoder(text_ids, attention_mask=text_masks)
        memory_input = torch.cat([vis_feat, txt_feat], dim=1)
        memory_input = model.pos_encoder(memory_input)
        
        # 构造 Encoder 掩码防污染
        num_vis = vis_feat.size(1)
        vis_pad_mask = torch.zeros((B, num_vis), dtype=torch.bool, device=device)
        txt_pad_mask = (text_masks == 0).bool()
        memory_key_padding_mask = torch.cat([vis_pad_mask, txt_pad_mask], dim=1)
        
        # 获取最终 Memory
        memory = model.transformer_encoder(memory_input, src_key_padding_mask=memory_key_padding_mask)
        
        # 2. 初始化 Decoder 输入板：所有 batch 都先放一个 <BOS>
        # 形状初始化为 [B, 1]
        decoder_input = torch.full((B, 1), self.BOS_TOKEN, dtype=torch.long, device=device)
        
        # 3. 自回归大循环 (一步一步往后猜)
        for _ in range(max_len - 1):
            # 将当前序列转为向量，并加位置编码
            tgt_embed = model.tgt_embedding(decoder_input)
            tgt_embed = model.pos_encoder(tgt_embed)
            
            # 生成下三角掩码，防止自己偷看自己
            #不同于训练时为了因果性（防止偷看未来），推理时是为了保护过去，防止影响到过去词的表示
            tgt_seq_len = decoder_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
            
            # Decoder 进行注意力交互
            #注意力输出的是什么？	每个 Query token 的“上下文加权”表示
            dec_out = model.transformer_decoder(
                tgt=tgt_embed, 
                memory=memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # 获取最后一个时间步（刚刚生成出来的那个词）的分类分布
            # dec_out 形状: [B, curr_len, 256]，我们只要最后一步: [:, -1, :]
            #[:,-1,:]代表第一个维度所有样本都取，第二个维度只取最后一个，第三个维度取全部
            last_step_features = dec_out[:, -1, :]
            logits = model.head(last_step_features) # [B, vocab_size]
            
            # 贪心解码：选概率最大的那个词
			'''
			# 假设 dec_out 形状为 [B, curr_len, 256]
			# 各维度索引
			dim=0   → 第1个维度 = B (batch)
			dim=1   → 第2个维度 = curr_len (序列长度)
			dim=2   → 第3个维度 = 256 (特征维度)
			dim=-1  → 最后一个维度 = 256 (等同于 dim=2)
			dim=-2  → 倒数第二个维度 = curr_len (等同于 dim=1)
			dim=-3  → 倒数第三个维度 = B (等同于 dim=0)
			'''
            next_word = torch.argmax(logits, dim=-1) # [B],dim=-1代表最后一个维度
            
            # 拼接到输入板后面，准备下一次循环
            #squeeze 挤压，unsqueeze（dim）在指定位置插入一个维度为1的新维度
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
            
        return decoder_input

    def extract_ptbb(self, generated_seq: list):
        """
        从模型生成的长序列中，扒出第二阶段的精确框 (PTBB)。
        """
        try:
            # 寻找 <SEP> 的位置
            sep_idx = generated_seq.index(self.SEP_TOKEN)
            # 取出 SEP 后面的 4 个数字,因为Python切片是左闭右开，包含起始索引，不包含结束最后一个索引
            ptbb_tokens = generated_seq[sep_idx+1 : sep_idx+5]
            return ptbb_tokens
        except ValueError:
            # 如果模型很笨，根本没生成 SEP，我们只能退而求其次，取最后 4 个数，或者返回空
            # 这里保守起见，如果没 SEP，说明格式彻底乱了
            return []