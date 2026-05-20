# 文件路径: test.py
import os
import json
import torch
from tqdm import tqdm  # 用于显示的进度条
from PIL import Image

# 导入我们的模块
from data.dataset import SkyFindDataset
from models.seqtr_aerial import AerialRECSeqTR
from engine.evaluator import AerialRECEvaluator

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 启动推理评测引擎，使用设备: {device}")
    
    # ==========================================
    # 1. 路径与配置
    # ==========================================
    VAL_JSON = "/root/group_data/Test.json"
    IMG_DIR = "/root/group_data/SkyFind-images"
    WEIGHTS_PATH = "checkpoints/best_model1.pth" # 黄金权重！
    
    # ==========================================
    # 2. 初始化核心组件
    # ==========================================
    print(" 初始化模型与评估器...")
    model = AerialRECSeqTR(hidden_dim=256, num_bins=1000).to(device)
    
    # 严格加载最好权重
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=True)
    model.eval() # 必须开启评估模式，冻结 Dropout 等
    
    # 初始化我们之前写的 Evaluator
    evaluator = AerialRECEvaluator(num_bins=1000)
    
    # 为了复用预处理逻辑，我们实例化 Dataset，但我们手写循环读取
    # 这样方便我们拿到原始图像的长宽 (img_w, img_h)
    dataset = SkyFindDataset(json_path=VAL_JSON, img_dir=IMG_DIR)
    
    # ==========================================
    # 3. 评测统计变量
    # ==========================================
    total_iou = 0.0
    correct_05 = 0
    valid_samples = 0
    
    print("\n 开始自回归逐词推理评测 (因为是一个词一个词蹦，速度较慢，请耐心观察)...")
    
    # 使用 tqdm 包装，显示进度条
    with torch.no_grad(): # 切断梯度，纯推理
        for idx in tqdm(range(len(dataset)), desc="Evaluating Validation Set"):
            # 1. 获取原始数据信息
            anno = dataset.annotations[idx]
            img_name = anno['fileName']
            gt_box = anno['bbox'] # 真实的 [x1, y1, x2, y2]
            
            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path):
                continue
                
            # 获取原始长宽，用于逆映射
            try:
                raw_image = Image.open(img_path).convert("RGB")
                img_w, img_h = raw_image.size
            except Exception:
                continue # 跳过损坏的图片
                
            # 2. 调用 dataset 获取张量 (借用它的 transform 和 tokenizer)
            sample = dataset[idx]
            images = sample['image'].unsqueeze(0).to(device) # 增加 Batch 维度: [1, 3, 640, 640]
            text_ids = sample['text_input_ids'].unsqueeze(0).to(device) # [1, 40]
            text_masks = sample['text_attention_mask'].unsqueeze(0).to(device) # [1, 40]
            
            # 3. 核心大招：自回归生成！
            # 输出类似: [[1000, 311, ... 1002, x1, y1, x2, y2, 1001]]
            pred_seq_tensor = evaluator.generate_sequence(model, images, text_ids, text_masks, max_len=11)
            pred_seq_list = pred_seq_tensor[0].cpu().tolist()
            
            # 4. 解析预测的序列，提取第二阶段 PTBB
            ptbb_tokens = evaluator.extract_ptbb(pred_seq_list)
            
            # 如果模型很笨没生成精确框，记 IoU=0
            if len(ptbb_tokens) != 4:
                iou = 0.0
            else:
                # 逆映射回浮点物理坐标
                pred_box = evaluator.detokenize_box(ptbb_tokens, img_w, img_h)
                # 计算几何 IoU
                iou = evaluator.box_iou(pred_box, gt_box)
            
            # 5. 累计指标
            total_iou += iou
            if iou >= 0.5:
                correct_05 += 1
            valid_samples += 1
            
    # ==========================================
    # 4. 推理评估
    # ==========================================
    if valid_samples > 0:
        iou_mean = (total_iou / valid_samples) * 100
        iou_05 = (correct_05 / valid_samples) * 100
        
        print("\n" + "="*50)
        print(" 评测报告 (Validation Results) ")
        print("="*50)
        print(f"总有效评测样本数 : {valid_samples}")
        print(f"IoU@mean (平均重合度) : {iou_mean:.2f} %")#这里是计算总的Iou@50平均
        print(f"IoU@0.5 (精准命中率)  : {iou_05:.2f} %")
        print("="*50)

if __name__ == "__main__":
    evaluate_model()