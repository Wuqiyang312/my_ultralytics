from ultralytics.nn.modules import Conv
import torch
import torch.nn as nn


class TextureMaker(nn.Module):
    """
    纹理生成模块（早期融合用）
    作用：在 RGB 图像基础上，增加 Sobel X/Y 方向的纹理特征通道，并调整到指定输出通道数
    """
    def __init__(self, out_ch=3):
        super().__init__()
        # Sobel 卷积核
        self.sobel_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

        sobel_x_k = torch.tensor(
            [[[[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]]],
            dtype=torch.float32
        )
        sobel_y_k = torch.tensor(
            [[[[-1, -2, -1],
               [0,  0,  0],
               [1,  2,  1]]]],
            dtype=torch.float32
        )
        self.sobel_x.weight.data.copy_(sobel_x_k)
        self.sobel_y.weight.data.copy_(sobel_y_k)

        # 输出调整到 out_ch
        self.adjust = nn.Conv2d(5, out_ch, 1, 1, 0)

    def forward(self, x):
        # 转灰度
        gray = x.mean(1, keepdim=True)
        # Sobel X/Y
        tx = self.sobel_x(gray)
        ty = self.sobel_y(gray)
        # 拼接 RGB + 纹理
        fused = torch.cat([x, tx, ty], 1)
        # 调整通道
        return self.adjust(fused)


class TextureStem(nn.Module):
    """
    纹理主干模块（中期融合用）
    作用：对输入纹理图像进行两次下采样卷积提取特征
    """
    def __init__(self, out_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(3, out_ch, 3, 2),   # 与 YOLO 主干一致的 Conv 模块
            Conv(out_ch, out_ch, 3, 2)
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    # 测试早期融合
    tm = TextureMaker(out_ch=3)
    img = torch.randn(1, 3, 640, 640)  # 假设输入 RGB
    out_tm = tm(img)
    print("TextureMaker 输出形状:", out_tm.shape)

    # 测试中期融合
    ts = TextureStem(out_ch=64)
    tex_img = torch.randn(1, 3, 640, 640)
    out_ts = ts(tex_img)
    print("TextureStem 输出形状:", out_ts.shape)
