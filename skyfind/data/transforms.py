# 文件路径: data/transforms.py
import torch
import torchvision.transforms.functional as F
from PIL import Image

class AerialRECTransform:
    """
    针对 AerialREC 任务的数据变换器。
    包含图像缩放、张量化、标准化，以及核心的坐标离散化与序列拼接。
    """
    def __init__(self, img_size: int = 640, num_bins: int = 1000):
        """
        :param img_size: 统一的图像输入尺寸 (默认 640x640)
        :param num_bins: 坐标离散化的桶数量 (默认 1000，即把图像整体坐标分成 1000 份)
        """
        self.img_size = img_size
        self.num_bins = num_bins
        
        # 定义特殊 Token 的 ID (紧跟在坐标 bins 之后)
        # 例如 bins 的范围是 0 ~ 999
        self.BOS_TOKEN = num_bins       # 1000
        self.EOS_TOKEN = num_bins + 1   # 1001
        self.SEP_TOKEN = num_bins + 2   # 1002
        self.PAD_TOKEN = num_bins + 3   # 1003 (用于后续 DataLoader 对齐)
        self.vocab_size = num_bins + 4  # 模型分类头的类别总数: 1004

    def _discretize(self, coord: float, max_val: float) -> int:
        """
        核心机理：将连续坐标转换为离散的 Token ID。
        """
        # 防止越界
        coord = max(0.0, min(coord, max_val))
        # 按比例映射到 [0, num_bins-1]
        token_id = int((coord / max_val) * (self.num_bins - 1))
        return token_id

    def __call__(self, image: Image.Image, ptr_box: list, ptbb_box: list):
        """
        前向处理单条数据
        """
        orig_w, orig_h = image.size

        # 1. 图像变换 (Resize -> ToTensor -> Normalize)
        image = F.resize(image, (self.img_size, self.img_size))
        image = F.to_tensor(image)
        #！！！为什么要标准化？
        # 使用标准的 ImageNet 均值和方差
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 2. 坐标等比例离散化 (注意：x对应宽度w，y对应高度h)
        ptr_x1 = self._discretize(ptr_box[0], orig_w)
        ptr_y1 = self._discretize(ptr_box[1], orig_h)
        ptr_x2 = self._discretize(ptr_box[2], orig_w)
        ptr_y2 = self._discretize(ptr_box[3], orig_h)

        ptbb_x1 = self._discretize(ptbb_box[0], orig_w)
        ptbb_y1 = self._discretize(ptbb_box[1], orig_h)
        ptbb_x2 = self._discretize(ptbb_box[2], orig_w)
        ptbb_y2 = self._discretize(ptbb_box[3], orig_h)

        # 3. 按照 AerialREC 的逻辑组装目标序列 (Target Sequence)
        # 序列格式: <BOS> ptr_x1 ptr_y1 ptr_x2 ptr_y2 <SEP> ptbb_x1 ptbb_y1 ptbb_x2 ptbb_y2 <EOS>
        target_seq =[
            self.BOS_TOKEN,
            ptr_x1, ptr_y1, ptr_x2, ptr_y2,
            self.SEP_TOKEN,
            ptbb_x1, ptbb_y1, ptbb_x2, ptbb_y2,
            self.EOS_TOKEN
        ]
        
        target_tensor = torch.tensor(target_seq, dtype=torch.long)#！！！64位整数，后续显存不够来这里改

        return image, target_tensor

# ==========================================
# 交互与验证环节：看看连续坐标怎么变成离散 Token
# ==========================================
if __name__ == "__main__":
    # 模拟一张图片和刚刚你跑出来的数据
    dummy_img = Image.new('RGB', (1360, 765))
    ptr =[397.89, 482.68, 847.57, 693.14]
    ptbb =[424, 507, 516, 644]
    
    transform = AerialRECTransform()
    img_tensor, seq_tensor = transform(dummy_img, ptr, ptbb)
    
    print("="*40)
    print("Transform 模块验证成功！")
    print(f"转换后的图像 Tensor 形状: {img_tensor.shape}")
    print(f"转换后的坐标 Tensor 形状: {seq_tensor.shape}")
    print(f"离散化后目标序列 Token IDs: \n{seq_tensor.tolist()}")
    print("序列含义对照:")
    print(f"BOS({transform.BOS_TOKEN}) | "
          f"PTR({seq_tensor[1:5].tolist()}) | "
          f"SEP({transform.SEP_TOKEN}) | "
          f"PTBB({seq_tensor[6:10].tolist()}) | "
          f"EOS({transform.EOS_TOKEN})")
    print("="*40)