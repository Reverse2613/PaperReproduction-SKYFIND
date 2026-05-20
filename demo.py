# 文件路径: demo.py
import os
import random
import torch
from PIL import Image, ImageDraw
from data.dataset import SkyFindDataset
from models.seqtr_aerial import AerialRECSeqTR
from engine.evaluator import AerialRECEvaluator

def run_demo(num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 可视化分析，使用设备: {device}")
    
    # ==========================================
    # 1. 路径与配置
    # ==========================================
    VAL_JSON = "/root/group_data/Val.json"
    IMG_DIR = "/root/group_data/SkyFind-images"
    WEIGHTS_PATH = "checkpoints/best_model1.pth" 
    OUTPUT_DIR = "demo_results/" # 画好框的图片保存在这里
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==========================================
    # 2. 初始化模型与数据集
    # ==========================================
    print(" 加载模型与黄金权重...")
    model = AerialRECSeqTR(hidden_dim=256, num_bins=1000).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=True)
    model.eval() 
    
    evaluator = AerialRECEvaluator(num_bins=1000)
    dataset = SkyFindDataset(json_path=VAL_JSON, img_dir=IMG_DIR)
    
    # 随机抽取 num_samples 张图
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    print("\n" + "="*60)
    print(" 开始逐图推理与坐标对比")
    print("="*60)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # 获取原始信息
            anno = dataset.annotations[idx]
            img_name = anno['fileName']
            expression = anno['expression']
            gt_box = anno['bbox'] # 真实框
            
            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path):
                continue
                
            raw_image = Image.open(img_path).convert("RGB")
            img_w, img_h = raw_image.size
            
            # 获取网络输入张量
            sample = dataset[idx]
            images = sample['image'].unsqueeze(0).to(device)
            text_ids = sample['text_input_ids'].unsqueeze(0).to(device)
            text_masks = sample['text_attention_mask'].unsqueeze(0).to(device)
            
            # 模型自回归预测
            pred_seq_tensor = evaluator.generate_sequence(model, images, text_ids, text_masks, max_len=11)
            pred_seq_list = pred_seq_tensor[0].cpu().tolist()#！！！
            
            # 提取与逆映射精确框
            ptbb_tokens = evaluator.extract_ptbb(pred_seq_list)
            if len(ptbb_tokens) == 4:
                pred_box = evaluator.detokenize_box(ptbb_tokens, img_w, img_h)
                iou = evaluator.box_iou(pred_box, gt_box)
            else:
                pred_box = [0.0, 0.0, 0.0, 0.0]
                iou = 0.0
                
            # ------------------------------------------
            # 终端打印对比
            # ------------------------------------------
            print(f"\n[{i+1}/{num_samples}] 图片文件: {img_name}")
            print(f" 文本指令: {expression}")
            print(f" 真实坐标 (GT)  : {[round(x, 1) for x in gt_box]}")
            print(f" 预测坐标 (Pred): {[round(x, 1) for x in pred_box]}")
            print(f" 几何交并比 (IoU): {iou:.4f} " + ("(命中!)" if iou >= 0.5 else "(偏离)"))
            
            # ------------------------------------------
            # 图像绘制并保存
            # ------------------------------------------
            # 创建画笔
            draw = ImageDraw.Draw(raw_image)
            # 红色代表真实的 Ground Truth，线宽为 3
            draw.rectangle(gt_box, outline="red", width=3)
            # 绿色代表模型的 Prediction，线宽为 3
            draw.rectangle(pred_box, outline="lime", width=3)
            
            save_path = os.path.join(OUTPUT_DIR, f"demo_{i+1}_{img_name}")
            raw_image.save(save_path)
            
    print("\n" + "="*60)
    print(f" 所有可视化结果已保存至文件夹: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_demo(num_samples=5)