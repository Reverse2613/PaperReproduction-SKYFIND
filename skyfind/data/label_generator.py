# 文件路径：data/label_generator.py
import random
from typing import Tuple,List

class AerialRECLabelGenerator:
    """基于SkyFind paper复现的标签生成器
    生成扩大范围的潜在目标区域，平滑优化过程的损失曲面
    缓解无人机视角下小目标带来的优化梯度陡峭问题
    """
    def __init__(self,alpha:float=0.4):
        """
        初始化标签生成器。
        :param alpha: 控制扩大区间大小的超参数。论文(Section 5.1)指出 alpha=0.4 性能最佳。
        """
        self.alpha = alpha
    
    #！！！
    def _safe_clip(self, value: float, min_val: float, max_val: float) -> float:
        """
        安全防护：确保生成的坐标绝对不会越界（图像的物理边界限制）。
        将value限制在[min_val,max_val]范围内
        """
        return max(min_val, min(value, max_val))

    def generate_two_stage_boxes(
        self, 
        gt_box: Tuple[float, float, float, float], 
        img_width: float, 
        img_height: float
    ) -> Tuple[List[float], List[float]]:
        """
        生成两阶段的坐标：PTR (潜在目标区域) 和 PTBB (精准目标边界框)
        
        :param gt_box: Ground Truth BBox, 格式为 (x1, y1, x2, y2)，绝对像素坐标
        :param img_width: 图像宽度
        :param img_height: 图像高度
        :return: (ptr_box, ptbb_box)
        """
        x1, y1, x2, y2 = gt_box

        # 公式 (4): 计算目标框外围的剩余空间
        x_left = img_width - (x2 - x1)
        y_left = img_height - (y2 - y1)

        # 公式 (5): 计算左上角点 (x1, y1) 的采样范围 R1
        # R1_x \in[max(0, x1 - alpha * x_left), x1]
        # R1_y \in[max(0, y1 - alpha * y_left), y1]
        r1_x_min = max(0.0, x1 - self.alpha * x_left)
        r1_x_max = x1
        r1_y_min = max(0.0, y1 - self.alpha * y_left)
        r1_y_max = y1

        # 公式 (6): 计算右下角点 (x2, y2) 的采样范围 R2
        # R2_x \in [x2, min(w, x2 + alpha * x_left)]
        # R2_y \in [y2, min(h, y2 + alpha * y_left)]
        r2_x_min = x2
        r2_x_max = min(img_width, x2 + self.alpha * x_left)
        r2_y_min = y2
        r2_y_max = min(img_height, y2 + self.alpha * y_left)

        # 公式 (7) & (8): 在空间分布中随机采样，产生平滑损失曲面的效果
        ptr_x1 = random.uniform(r1_x_min, r1_x_max)
        ptr_y1 = random.uniform(r1_y_min, r1_y_max)
        ptr_x2 = random.uniform(r2_x_min, r2_x_max)
        ptr_y2 = random.uniform(r2_y_min, r2_y_max)

        # 最终安全检查，防止浮点数精度或极端长宽比导致越界
        
        #！！！
        #将左上角的点坐标限制在图像范围内，[0,img_width]和[0,img_height]之间
        ptr_x1 = self._safe_clip(ptr_x1, 0, img_width)
        ptr_y1 = self._safe_clip(ptr_y1, 0, img_height)
        #将右下角的点坐标限制在图像范围内，并且确保它们不小于左上角的坐标
        #[x1,img_width]和[y1,img_height]之间
        ptr_x2 = self._safe_clip(ptr_x2, ptr_x1, img_width)
        ptr_y2 = self._safe_clip(ptr_y2, ptr_y1, img_height)

        ptr_box =[ptr_x1, ptr_y1, ptr_x2, ptr_y2]
        ptbb_box =[x1, y1, x2, y2] # 第二阶段回归真实精确坐标

        return ptr_box, ptbb_box

# ==========================================
# 验证代码 (仅供模块内部测试，不参与外部调用)
# ==========================================
if __name__ == "__main__":
    # 模拟一个 1920x1080 的无人机图像，其中包含一个极小的车辆目标
    img_w, img_h = 1920.0, 1080.0
    
    small_target_gt = (200.0, 500.0, 210.0, 510.0) 
    
    generator = AerialRECLabelGenerator(alpha=0.4)
    ptr, ptbb = generator.generate_two_stage_boxes(small_target_gt, img_w, img_h)
    
    print(f"Original Target Box (PTBB): {ptbb}")
    print(f"Sampled Potential Region (PTR): {ptr}")
    
    # 简单的安全性与逻辑验证 (验证阶段)
    assert ptr[0] <= ptbb[0], "PTR 左上角 x 必须包含 GT"
    assert ptr[1] <= ptbb[1], "PTR 左上角 y 必须包含 GT"
    assert ptr[2] >= ptbb[2], "PTR 右下角 x 必须包含 GT"
    assert ptr[3] >= ptbb[3], "PTR 右下角 y 必须包含 GT"
    print(" 验证通过：PTR 完全包裹了 PTBB，符合由粗到细的物理逻辑。")