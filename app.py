# 文件路径: app.py
import os
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

from models.seqtr_aerial import AerialRECSeqTR
from engine.evaluator import AerialRECEvaluator

def interactive_predict(image_path: str, text_prompt: str, weights_path: str):
    """
    交互式推理引擎
    给定任意一张图片和一句话，模型将进行理解并画出预测框。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 启动无人机交互视觉系统，使用设备: {device}")
    
    # ==========================================
    # 1. 验证文件是否存在
    # ==========================================
    if not os.path.exists(image_path):
        raise FileNotFoundError(f" 找不到图片文件: {image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f" 找不到权重文件: {weights_path}")

    # ==========================================
    # 2. 初始化模型、评估器和分词器
    # ==========================================
    print(" 正在加载模型...")
    model = AerialRECSeqTR(hidden_dim=256, num_bins=1000).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
    model.eval() # 切换到推理模式
    
    evaluator = AerialRECEvaluator(num_bins=1000)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ==========================================
    # 3. 模拟摄像头的图像预处理 (无需真实框)
    # ==========================================
    raw_image = Image.open(image_path).convert("RGB")
    img_w, img_h = raw_image.size
    
    # 与训练时完全一致的图像变换：Resize -> ToTensor -> Normalize
    img_tensor = F.resize(raw_image, (640, 640))
    img_tensor = F.to_tensor(img_tensor)
    img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 增加 Batch 维度: [3, 640, 640] -> [1, 3, 640, 640]
    images = img_tensor.unsqueeze(0).to(device)

    # ==========================================
    # 4. 模拟语音接收的文本预处理
    # ==========================================
    print(f" 发送指令: '{text_prompt}'")
    text_tokens = tokenizer(
        text_prompt, 
        padding='max_length', 
        truncation=True, 
        max_length=70, 
        return_tensors="pt"
    )
    text_ids = text_tokens['input_ids'].to(device)         # [1, 40]
    text_masks = text_tokens['attention_mask'].to(device)  # [1, 40]

    # ==========================================
    # 5. 开启自回归推理！
    # ==========================================
    print(" 模型正在思考与视觉搜索...")
    with torch.no_grad():
        pred_seq_tensor = evaluator.generate_sequence(model, images, text_ids, text_masks, max_len=11)
        pred_seq_list = pred_seq_tensor[0].cpu().tolist()#！！！
        
    print(f" 原始输出 Token 序列: {pred_seq_list}")

    # ==========================================
    # 6. 解析并画图
    # ==========================================
    ptbb_tokens = evaluator.extract_ptbb(pred_seq_list)
    
    if len(ptbb_tokens) != 4:
        print(" 模型理解失败或未找到目标！")
        return
        
    # 将离散 Token 转为物理坐标
    pred_box = evaluator.detokenize_box(ptbb_tokens, img_w, img_h)
    print(f" 最终锁定目标坐标: {[round(x, 1) for x in pred_box]}")#！！！
    
    # 开始在原图上画框
    draw = ImageDraw.Draw(raw_image)
    # 画一个亮绿色的预测框
    draw.rectangle(pred_box, outline="lime", width=4)
    
    # 尝试在框旁边写上文本提示 (增加科技感)
    try:
        # 如果系统有默认字体则使用，否则忽略
        draw.text((pred_box[0], max(0, pred_box[1]-15)), "Target Locked", fill="lime")
    except Exception:
        pass

    # 保存图片
    output_path = "output.jpg"
    raw_image.save(output_path)
    print("\n" + "="*50)
    print(f" 视觉定位完成！图片已保存至: {output_path}")
    print("="*50)


if __name__ == "__main__":
    # ==========================================
    # 在这里填入你想测试的图片和句子！
    # ==========================================
    
    # 1. 随便挑一张图片的绝对路径
    MY_IMAGE_PATH = "0000158.jpg" 
    
    # 2. 结合那张图片，用英文写一句话描述你想要找的物体
    # 比如："The white car on the cross road near the intersection."
    print("输入待发送的指令")
    MY_PROMPT = input()
    
    # 3. 我们的黄金权重路径
    WEIGHTS_PATH = "checkpoints/best_model1.pth" 
    
    interactive_predict(
        image_path=MY_IMAGE_PATH,
        text_prompt=MY_PROMPT,
        weights_path=WEIGHTS_PATH
    )