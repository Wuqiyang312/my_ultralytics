# ultralytics/nn/modules/texture_modules.py
import torch
import torch.nn as nn


class TextureMaker(nn.Module):
    """
    纹理生成模块（早期融合）
    Ultralytics 解析时会调用 TextureMaker(c1, out_ch)，因此这里的签名需要兼容 (c1, out_ch)。
    逻辑：输入 x (B, c1, H, W)，计算灰度 -> Sobel X/Y -> 与原始输入拼接 -> 1x1 调整到 out_ch。
    """
    def __init__(self, c1=3, out_ch=3):
        super().__init__()
        self.c1 = c1
        # Sobel 卷积核（固定权重）
        self.sobel_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        sobel_x_k = torch.tensor([[[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]]]).float()
        sobel_y_k = torch.tensor([[[[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]]]]).float()
        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_x_k)
            self.sobel_y.weight.copy_(sobel_y_k)
        # 拼接后通道为 c1 + 2
        self.adjust = nn.Conv2d(c1 + 2, out_ch, 1, 1, 0)

    def forward(self, x):
        # 灰度（兼容任意 c1）
        gray = x.mean(1, keepdim=True)
        # Sobel 边缘
        tx = self.sobel_x(gray)
        ty = self.sobel_y(gray)
        # 拼接原图通道与边缘通道
        fused = torch.cat([x, tx, ty], 1)
        return self.adjust(fused)


class _FixedSobel(nn.Module):
    """固定 Sobel 提取，输出 2 通道 [gx, gy]"""
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        sobel_x_k = torch.tensor([[[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]]]).float()
        sobel_y_k = torch.tensor([[[[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]]]]).float()
        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_x_k)
            self.sobel_y.weight.copy_(sobel_y_k)

    def forward(self, x):
        gray = x.mean(1, keepdim=True)
        gx = self.sobel_x(gray)
        gy = self.sobel_y(gray)
        return torch.cat([gx, gy], 1)  # [B,2,H,W]


class _BlurPool2d(nn.Module):
    """
    轻量 Anti-aliasing 下采样，默认 3x3 avgpool, stride=2, padding=1
    """
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


class TextureStemLite(nn.Module):
    """
    轻量纹理干支（中期融合用）
    签名兼容 Ultralytics：TextureStemLite(c1, c2)
    主分支：Conv(c1->c2, k=3, s=2, p=1)
    边缘分支：固定 Sobel -> BlurPool2d 下采样 -> 1x1 映射到 c2
    输出：主分支 + 边缘分支（同尺度 H/2,W/2, 通道 c2）
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.main = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.edge = _FixedSobel()
        self.edge_down = _BlurPool2d(kernel_size=3, stride=2, padding=1)
        self.edge_proj = nn.Conv2d(2, c2, 1, bias=False)

    def forward(self, x):
        y = self.act(self.bn(self.main(x)))            # [B,c2,H/2,W/2]
        e = self.edge_proj(self.edge_down(self.edge(x)))  # [B,c2,H/2,W/2]
        return y + e