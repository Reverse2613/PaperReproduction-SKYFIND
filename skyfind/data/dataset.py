# 文件路径: data/dataset.py
import json
import os
import torch
from torch.utils.data import Dataset
import random
from PIL import Image,ImageFile
from transformers import AutoTokenizer # 使用 HuggingFace 分词器处理文本

from data.transforms import AerialRECTransform  # 引入写的 Transform
from data.label_generator import AerialRECLabelGenerator# 引入写好的标签生成器

ImageFile.LOAD_TRUNCATED_IMAGES = True # 允许处理轻微损坏的图像文件，防止 PIL 报错

class SkyFindDataset(Dataset):
    """
    SkyFind 数据集加载类
    严格按照单一职责原则，本类只负责：读取硬盘数据 -> 解析JSON -> 生成两阶段Box坐标
    """
    def __init__(self, json_path: str, img_dir: str, alpha: float = 0.4, img_size: int = 640):
        """
        :param json_path: 标注文件的路径 (例如: 'train.json')
        :param img_dir: 图片所在的文件夹路径 (例如: 'images/train/')
        :param alpha: AerialREC 两阶段坐标生成器的超参数 (论文默认为 0.4)
        """
        self.img_dir = img_dir
        
        # 1. 加载标注数据
        print(f"Loading annotations from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)#！！！
        print(f"Successfully loaded {len(self.annotations)} samples.")
            
        # 2. 初始化论文提出的两阶段标签生成器
        self.label_generator = AerialRECLabelGenerator(alpha=alpha)
        
        #3. 初始化 transform
        self.transform = AerialRECTransform(img_size = img_size, num_bins=1000)

        #4.初始化文本的分词器Tokenizer（使用bert-base-uncased作为基线标准）
        #初始运行会自动下载词表
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        # 1. 按照json文件中的真实字段名获取数据
        anno = self.annotations[idx]
        img_name = anno['fileName']       #  "fileName"
        expression = anno['expression']   #  "expression"
        gt_box = anno['bbox']             #  "bbox"[x_min, y_min, x_max, y_max]

        try:
            # 2. 读取图像
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
        

        
            # 3. 使用论文的方法，把单一的 GT Box 变成 [PTR, PTBB] 两阶段框
            ptr_box, ptbb_box = self.label_generator.generate_two_stage_boxes(
                gt_box=gt_box, 
                img_width=img_w, 
                img_height=img_h
            )
        
            # 4.图像与坐标的Transforms ，得到模型输入需要的Tensor格式
            img_tensor,target_seq = self.transform(image, ptr_box, ptbb_box)

            # 5.文本处理tokenizer，截断并填充
            text_tokens = self.tokenizer(
                expression, 
                padding='max_length', 
                truncation=True, 
                max_length=70, 
                return_tensors='pt'
            )

            # 6. 组装喂给模型的字典
            sample = {
                "image": img_tensor,           # 处理后的图像 Tensor,维度形状[3,640,640]
                "text_input_ids": text_tokens['input_ids'].squeeze(0), # ！！！文本的 token ids Tensor[70]
                "text_attention_mask": text_tokens['attention_mask'].squeeze(0), # ！！！文本的 attention mask Tensor[70]
                "target_seq": target_seq,       # 离散化后的目标序列 Tensor[12],11个Token ID
            }
        
            return sample

        except Exception as e:
            # 捕获到坏图，打印一行警告（
            print(f"\n 警告: 读取第 {idx} 个样本 (文件为 {img_name}) 时发生异常: {e}。已启动容错，随机替换样本...")
            # 随机挑选另一个索引，递归调用自己，直到拿到好图为止！
            new_idx = random.randint(0, len(self.annotations) - 1)
            return self.__getitem__(new_idx)
        

# ==========================================
# 交互与验证环节：打破黑盒，所见即所得
# ==========================================
if __name__ == "__main__":
    # 把下面这两个路径改成真实的路径
    # 例如：
    # TEST_JSON_PATH = "/root/group_data/Train.json"
    # TEST_IMG_DIR = "/root/group_data/SkyFind-images"
    
    TEST_JSON_PATH = "/root/group_data/Train.json" 
    TEST_IMG_DIR = "/root/group_data/SkyFind-images"
    
    try:
        # 实例化数据集
        test_dataset = SkyFindDataset(json_path=TEST_JSON_PATH, img_dir=TEST_IMG_DIR, alpha=0.4,img_size=640)
        
        # 取出第0个样本
        sample_0 = test_dataset[0]
        
        print("\n" + "="*40)
        print(" 数据集模块验证成功！")
        print("="*40)
        print(f"img_tensor,图像张量形状: {sample_0['image'].shape}")
        print(f"文本Input_IDS形状: {sample_0['text_input_ids'].shape}")
        print(f"文本Input_IDS形状: {sample_0['text_input_ids']}")
        print(f"目标序列张量: {sample_0['target_seq']}")
        print("="*40)
        
    except Exception as e:
        print(f"\n 验证失败，报错信息如下: {e}")
        print("请检查路径是否正确，或者图片是否存在。")