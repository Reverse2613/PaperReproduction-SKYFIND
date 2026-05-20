# 文件路径: analysis.py
import os
import torch
import torch.nn.functional as F_torch
import torchvision.transforms.functional as F_vis
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 # 需要安装 opencv-python

from transformers import AutoTokenizer
from models.seqtr_aerial import AerialRECSeqTR
from engine.evaluator import AerialRECEvaluator
from app import interactive_predict

def generate_attention_heatmap(image_path: str, text_prompt: str, weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 启动注意力可视化...")
    
    # 1. 初始化模型
    model = AerialRECSeqTR(hidden_dim=256, num_bins=1000).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
    model.eval() 
    
    evaluator = AerialRECEvaluator(num_bins=1000)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 2. 图像预处理
    raw_image = Image.open(image_path).convert("RGB")
    img_tensor = F_vis.resize(raw_image, (640, 640))
    img_tensor = F_vis.to_tensor(img_tensor)
    img_tensor = F_vis.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = img_tensor.unsqueeze(0).to(device)

    # 3. 文本预处理
    text_tokens = tokenizer(text_prompt, padding='max_length', truncation=True, max_length=40, return_tensors="pt")
    text_ids = text_tokens['input_ids'].to(device)
    text_masks = text_tokens['attention_mask'].to(device)

    print(" 正在提取深层特征矩阵...")
    with torch.no_grad():
        # A. 提取视觉特征 (400个网格的记忆)
        vis_feat = model.visual_encoder(images) # [1, 400, 256]
        txt_feat = model.text_encoder(text_ids, attention_mask=text_masks)
        
        memory_input = torch.cat([vis_feat, txt_feat], dim=1)
        memory_input = model.pos_encoder(memory_input)
        
        # 掩码防污染
        num_vis = vis_feat.size(1)
        vis_pad_mask = torch.zeros((1, num_vis), dtype=torch.bool, device=device)
        txt_pad_mask = (text_masks == 0).bool()
        memory_key_padding_mask = torch.cat([vis_pad_mask, txt_pad_mask], dim=1)
        
        # 完整的图文融合 Memory
        memory = model.transformer_encoder(memory_input, src_key_padding_mask=memory_key_padding_mask)
        
        # 我们只关心图像部分的 Memory (前 400 个)
        image_memory = memory[:, :400, :] # [1, 400, 256]

        # B. 自回归生成，并拦截中间特征
        decoder_input = torch.full((1, 1), evaluator.BOS_TOKEN, dtype=torch.long, device=device)
        
        # 用于记录每个阶段的注意力图
        attention_maps = {}
        
        for step in range(10): # 最多生成 10 个词
            tgt_embed = model.tgt_embedding(decoder_input)
            tgt_embed = model.pos_encoder(tgt_embed)
            tgt_seq_len = decoder_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)
            
            dec_out = model.transformer_decoder(
                tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask
            )
            
            # 当前这一步 Decoder 吐出的特征查询向量 Query
            current_query = dec_out[:, -1, :].unsqueeze(1) # [1, 1, 256]
            
            # 预测下一个词
            logits = model.head(current_query.squeeze(1))
            next_word = torch.argmax(logits, dim=-1).item()
            
            # 【核心数学计算：计算 Query 和 400个图像网格的内积相似度】
            # (1, 1, 256) x (1, 256, 400) -> (1, 1, 400)
            similarity = torch.bmm(current_query, image_memory.transpose(1, 2))
            
            # 缩放并归一化 (类似 Softmax)
            attn_scores = F_torch.softmax(similarity / (256 ** 0.5), dim=-1).squeeze() # [400]
            
            # 保存关键阶段的注意力图
            if step == 0:
                attention_maps['Stage1_PTR_Start'] = attn_scores
            elif next_word == evaluator.SEP_TOKEN:
                # 刚刚预测出了 SEP，下一步就要预测精细框了！
                # 此时的 Query 包含了第一阶段找出的粗略范围信息
                # 这是最暴露问题的一张图！
                attention_maps['Stage2_PTBB_Start'] = attn_scores
                
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == evaluator.EOS_TOKEN:
                break

    # ==========================================
    # 4. 绘制并叠加伪彩色热力图
    # ==========================================
    print("正在渲染热力图...")
    raw_img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    
    for stage_name, attn_tensor in attention_maps.items():
        # [400] -> [20, 20]
        attn_grid = attn_tensor.view(20, 20).cpu().numpy()
        
        # 归一化到 0 ~ 255
        attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
        attn_grid = np.uint8(255 * attn_grid)
        
        # 双线性插值放大到原图大小
        heatmap = cv2.resize(attn_grid, (raw_img_cv.shape[1], raw_img_cv.shape[0]))
        
        # 涂上 JET 伪彩色 (红色最高，蓝色最低)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 将热力图按 0.5 的透明度盖在原图上
        overlay = cv2.addWeighted(raw_img_cv, 0.5, heatmap, 0.5, 0)
        
        # 保存
        output_name = f"attn_{stage_name}.jpg"
        cv2.imwrite(output_name, overlay)
        print(f" 成功保存特征热力图: {output_name}")

if __name__ == "__main__":
    # 如果没有 cv2，请在终端运行: pip install opencv-python
    WEIGHTS_PATH = "checkpoints/best_model1.pth" 
    # 找一张有多个相似物体干扰的图，比如停车场、一群人
    TEST_IMG = "0000158.jpg"
    print("输入指令")
    TEST_PROMPT = input()
    
    interactive_predict(
        image_path=TEST_IMG,
        text_prompt=TEST_PROMPT,
        weights_path=WEIGHTS_PATH
    )
    generate_attention_heatmap(TEST_IMG, TEST_PROMPT, WEIGHTS_PATH)