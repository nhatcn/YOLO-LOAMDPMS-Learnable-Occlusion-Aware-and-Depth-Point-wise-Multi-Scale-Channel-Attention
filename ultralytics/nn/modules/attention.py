import torch
import torch.nn as nn

class LOAMBlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, **kwargs):
        super().__init__()
        self.mask_gen = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, c1 // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1 // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1 // 4),
            nn.ReLU(inplace=True)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj = nn.Conv2d(c1 + c1 // 4, c2, kernel_size=1) if (c1 + c1 // 4) != c2 else nn.Identity()

    def forward(self, x):
        mask = self.mask_gen(x)
        enhanced = self.feature_enhance(x)
        out = x * (1 - self.alpha * mask)
        out = torch.cat([out, enhanced], dim=1)
        return self.proj(out)

class DPMS(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=1)
        self.conv2_dw = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1)
        self.conv2_pw = nn.Conv2d(c1, c1, kernel_size=1)
        self.conv3_dw = nn.Conv2d(c1, c1, kernel_size=5, padding=2, groups=c1)
        self.conv3_pw = nn.Conv2d(c1, c1, kernel_size=1)
        self.conv4_dw = nn.Conv2d(c1, c1, kernel_size=7, padding=3, groups=c1)
        self.conv4_pw = nn.Conv2d(c1, c1, kernel_size=1)
        self.fuse = nn.Conv2d(c1 * 4, c2, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // 16, c2, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Identity()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2_pw(self.conv2_dw(x))
        out3 = self.conv3_pw(self.conv3_dw(x))
        out4 = self.conv4_pw(self.conv4_dw(x))
        fused = self.fuse(torch.cat([out1, out2, out3, out4], dim=1))
        ca = self.channel_attention(fused)
        return fused * ca
