# 文件路径: train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast,GradScaler  #引入AMP自动混合精度加速训练 

# 导入我们的模块
from data.dataset import SkyFindDataset
from models.seqtr_aerial import AerialRECSeqTR

def train_engine():
    # ==========================================
    # 1. 硬件与超参数配置 (Hyper-parameters)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 启动训练引擎，使用设备: {device}")
    
    # 路径配置 (请替换为你的真实路径)
    TRAIN_JSON = "/root/group_data/Train.json" 
    VAL_JSON = "/root/group_data/Val.json"   #新增加验证集，来保存最佳模型权重
    IMG_DIR = "/root/group_data/SkyFind-images"
    SAVE_DIR = "checkpoints/" # 权重保存文件夹
    OLD_WEIGHTS_PATH = "checkpoints/best_model.pth" # 引入之前训练的权重文件
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 训练超参数
    BATCH_SIZE = 64    #  L20 有 48G 显存
    NUM_EPOCHS = 30    # 暂时定 30 轮
    MAX_LR = 1e-4
    WARMUP_EPOCHS = 3 # 学习率预热轮数，前3轮逐渐升高学习率，帮助模型稳定收敛,修复引入的旧权重坏习惯（学习到不好的东西）
    NUM_WORKERS = 8    #  CPU 有 10 核，开 8 个子进程加速数据读取

    # ==========================================
    # 2. 数据加载 (Data Loading)
    # ==========================================
    print(" 正在构建 Dataloader...")
    train_dataset = SkyFindDataset(json_path=TRAIN_JSON, img_dir=IMG_DIR)
    val_dataset = SkyFindDataset(json_path=VAL_JSON, img_dir=IMG_DIR)    
    # DataLoader 负责把单条数据打包成 Batch，并放入多进程加速
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,          # 全量训练必须打乱数据！打破样本间的时间相关性
        num_workers=NUM_WORKERS,
        pin_memory=True,       # 锁页内存，加速 CPU 到 GPU 的数据传输
        drop_last=True         # 丢弃最后一个不完整的 Batch，防止维度报错
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,          # 验证集不需要打乱，保持固定顺序方便调试和对比
        num_workers=NUM_WORKERS,
        pin_memory=True,       # 锁页内存，加速 CPU 到 GPU 的数据传输
    )
    print(f" Dataloader 就绪。训练批次 {len(train_loader)} 个 Batch，验证批次 {len(val_loader)} 个 Batch。")

    # ==========================================
    # 3. 模型、旧权重导入、优化器与损失函数
    # ==========================================
    model = AerialRECSeqTR(hidden_dim=256, num_bins=1000).to(device)

    #导入旧权重（如果存在的话），继续训练而不是从头开始
    if os.path.exists(OLD_WEIGHTS_PATH):
        print(f" 发现旧权重文件 {OLD_WEIGHTS_PATH}，正在加载...")
        #严格模式加载，确保权重完全匹配模型结构，如果不匹配会直接报错，避免引入不兼容的权重导致训练崩溃或性能下降
        model.load_state_dict(torch.load(OLD_WEIGHTS_PATH, map_location=device),strict=True)
        print(" 权重加载成功！准备开始训练...")
    else:
        print(" 没有找到旧权重文件，正在从头开始训练...")

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)

    # 【学习率调度器：Warmup + Cosine】
    # 这是一个极其重要的机制：防止加载旧权重后，面对新掩码突然崩溃。
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        return float(0.5 * (1.0 + torch.cos(torch.tensor((epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS) * 3.1415926))))
    # ！！！必须使用 LambdaLR，千万不要用 CosineAnnealingLR ！！！
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    '''这里的PAD_TOKEN_ID 只和计算损失函数有关，是连续坐标序列中的特殊标记，不是文本输入的 PAD_TOKEN_ID
    因为本次实验都是单图单目标（而且本次实验根本就没有补齐目标序列），没涉及到单图多目标，可以不加这个损失函数的 ignore_index了
    后续如果要加多目标检测，记得把这个 PAD_TOKEN_ID 加回去，确保模型不学习填充部分的损失。
    ignore_index=PAD_TOKEN_ID表示，当某个位置的token数值是PAD_TOKEN_ID时，
    CrossEntropyLoss会忽略这个位置的损失计算，这个位置的误差强制设为0，误差反向传播就不起作用，不会对模型参数产生梯度更新。
    这是因为填充部分不包含有效信息，不需要模型去学习。
    '''
    PAD_TOKEN_ID = 1003
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    #AMP加速，初始化梯度缩放器
    scaler = GradScaler('cuda') #如果使用GPU则启用混合精度训练，否则使用普通精度

    # ==========================================
    # 4. 安全核心：训练、验证循环与权重管理
    # ==========================================
    best_val_loss = float('inf') # 记录历史最低 Loss
    
    for epoch in range(NUM_EPOCHS):
        #----------------训练阶段----------------
        model.train()
        total_train_loss = 0.0
        
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] (LR: {scheduler.get_last_lr()[0]:.6f}) ---")
        
        # 遍历数据集
        for batch_idx, batch in enumerate(train_loader):
            # 将数据推送到显卡
            images = batch['image'].to(device)
            text_ids = batch['text_input_ids'].to(device)
            text_masks = batch['text_attention_mask'].to(device)
            tgt_seq = batch['target_seq'].to(device)
            #梯度清零
            optimizer.zero_grad()

            # 开启AMP半精度前向传播
            with autocast('cuda'): #启用自动混合精度
                logits = model(images,text_ids,text_masks,tgt_seq)
                # 错位计算 Loss (机理和 overfit 脚本完全一样)
                logits_flat = logits.view(-1, 1004)
                target_flat = tgt_seq[:,1:].contiguous().view(-1)
                loss = criterion(logits_flat, target_flat)
            
            # AMP反向传播与优化
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) #解码梯度，以便进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update() #更新梯度缩放器

            #累加训练损失
            total_train_loss += loss.item()
            
            # 打印进度条 (每 100 个 Batch 打印一次，避免刷屏)
            if (batch_idx + 1) % 100 == 0:
                print(f"  [Train Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        # Epoch 结束，更新学习率
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        #----------------验证阶段----------------
        model.eval() # 切换到评估模式，关闭 Dropout 和 BatchNorm
        total_val_loss = 0.0
        print("正在评估验证集")
        with torch.no_grad(): # 验证阶段绝不能计算梯度，省内存省算力！
            for batch in val_loader:
                images = batch['image'].to(device)
                text_ids = batch['text_input_ids'].to(device)
                text_masks = batch['text_attention_mask'].to(device)
                tgt_seq = batch['target_seq'].to(device)
                
                # 验证集也走 Teacher Forcing 前向传播算 Loss
                with autocast('cuda'):
                    logits = model(images, text_ids, text_masks, tgt_seq)
                    logits_flat = logits.view(-1, 1004)
                    target_flat = tgt_seq[:, 1:].contiguous().view(-1)
                    val_loss = criterion(logits_flat, target_flat)
                
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f" Epoch [{epoch+1}] 结束 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        # ==========================================
        #  硬盘保命机制：覆盖式保存
        # ==========================================
        # 1. 保存最新权重 (覆盖旧的 latest_model1.pth)
        latest_path = os.path.join(SAVE_DIR, "latest_model1.pth")
        torch.save(model.state_dict(), latest_path)
        print(f" 已保存最新权重至: {latest_path}")
        
        # 2. 如果当前 验证集Loss 是历史最低，则覆盖保存 best_model1.pth
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(SAVE_DIR, "best_model1.pth")
            torch.save(model.state_dict(), best_path)
            print(f" 发现更低 Loss ({best_val_loss:.4f})! 已覆盖保存最优权重至: {best_path}")

if __name__ == "__main__":
    # 执行前，强烈建议先不要跑全量！
    train_engine()