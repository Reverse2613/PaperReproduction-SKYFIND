# 文件路径: train_overfit.py
import torch
import torch.nn as nn
import torch.optim as optim

# 引入我们写好的数据集和模型
from models.seqtr_aerial import AerialRECSeqTR
from data.dataset import SkyFindDataset
from torch.utils.data import DataLoader

#！！！
def overfit_single_batch():
    """
    单批次过拟合测试 (Sanity Check)。
    科研准则：如果在 1 个 Batch 上 Loss 降不下去，说明模型架构或代码逻辑存在严重 Bug。
    绝不能在没通过该测试前去跑全量数据！
    """
    # 1. 硬件准备 (调用你的 L20 显卡)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 使用设备: {device}")

    # ==========================================
    # 把下面两个路径替换为数据存放的真实路径
    # ==========================================
    TEST_JSON_PATH = "/root/group_data/Train.json" 
    TEST_IMG_DIR = "/root/group_data/SkyFind-images"
    
    # 2. 准备数据 (只取 2 条数据作为一个 Batch)
    dataset = SkyFindDataset(json_path=TEST_JSON_PATH, img_dir=TEST_IMG_DIR)
    # shuffle=False 确保每次取的都是固定的最前面那2张图
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False) 
    
    # 获取唯一的这个 batch
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    text_ids = batch['text_input_ids'].to(device)
    text_masks = batch['text_attention_mask'].to(device)
    tgt_seq = batch['target_seq'].to(device)

    # 3. 初始化模型并放到显卡上
    # hidden_dim=256, num_bins=1000
    model = AerialRECSeqTR(hidden_dim=256, num_bins=1000).to(device)
    model.train() # 开启训练模式 (启用 Dropout, BatchNorm 等)

    # 4. 定义优化器和损失函数
    # 使用 AdamW，学习率设为 1e-4，这是 Transformer 常用的稳妥配置
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # CrossEntropyLoss: 忽略 PAD_TOKEN (1003) 的损失计算，因为填充部分不需要学习
    # 在 transforms.py 中我们定义了 PAD_TOKEN = 1003
    PAD_TOKEN_ID = 1003
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    print("\n 开始单批次过拟合测试 (迭代 100 次)...")
    
    # 5. 核心训练循环 (迭代 100 次，只看这一个 Batch)
    for epoch in range(100):
        optimizer.zero_grad() # 清空上一轮的旧梯度

        # 前向传播
        # 输出 logits 形状: [B, 10, 1004]
        logits = model(images, text_ids, text_masks, tgt_seq)

        # 错位机制 (Auto-regressive Shift) 计算 Loss
        # 1. logits 的形状需要变成 2D [B * 10, 1004]，因为 CrossEntropy 要求输入是 2D
        logits_flat = logits.view(-1, 1004)
        
        # 2. 真实标签 tgt_seq 的形状是 [B, 11] ([BOS, x1, y1, SEP, x2, y2, EOS])
        # 我们要去掉第 0 个词 (BOS)，取出 [x1, y1, SEP, x2, y2, EOS]，长度变成 10
        # 然后压平为 1D [B * 10]
        target_flat = tgt_seq[:, 1:].contiguous().view(-1)

        # 计算交叉熵损失
        loss = criterion(logits_flat, target_flat)

        # 反向传播，计算每个参数的梯度
        loss.backward()
        
        # 梯度裁剪 (Gradient Clipping)：防止 Transformer 训练早期梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器更新模型权重
        optimizer.step()

        # 每 10 次打印一次进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/100] | Loss: {loss.item():.4f}")
            
    print("\n 测试结束。如果最后 Loss 下降到了 0.5 以下，说明我们的代码架构没有大问题！")

if __name__ == "__main__":
    overfit_single_batch()