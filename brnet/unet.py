import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    
    def forward(self, x):
        batch, channels, _ = x.size()
        y = x.mean(dim=-1)  # Global average pooling
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = torch.sigmoid(y).view(batch, channels, 1)
        return y*x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_list):
        super().__init__()
        self.modules_list = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool1d(2),
                DoubleConv(in_channels, out_channels, kernel_size=kernel)
            ) for kernel in kernel_list
            ])

    def forward(self, x):
        # outputs = [module(x) for module in self.modules_list]
        return sum(module(x) for module in self.modules_list)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, weight):
        super().__init__()
        
        self.weight = weight
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        if weight == 0:
            self.conv = DoubleConv(in_channels//2, out_channels, in_channels // 2)
        else:
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # self.se_block = SEBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])
        # # if you have padding issues, see
        # # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.weight == 1:
            x1 = torch.cat([x2 * self.weight, x1], dim=1)
        # x1 = x1 * 0.7 + x2 * 0.3
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet1d(nn.Module):
    def __init__(self, n_channels, n_classes, nfilter=24, kernel_list = [3], up_weight = [1,1,1,1]):
        super(UNet1d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, nfilter)
        self.down1 = Down(nfilter, nfilter * 2, kernel_list)
        self.down2 = Down(nfilter * 2, nfilter * 4, kernel_list)
        self.down3 = Down(nfilter * 4, nfilter * 8, kernel_list)
        self.down4 = Down(nfilter * 8, nfilter * 8, kernel_list)
        self.up1 = Up(nfilter * 16, nfilter * 4, up_weight[0])
        self.up2 = Up(nfilter * 8, nfilter * 2, up_weight[1])
        self.up3 = Up(nfilter * 4, nfilter * 1, up_weight[2])
        self.up4 = Up(nfilter * 2, nfilter, up_weight[3])
        self.outc = OutConv(nfilter, n_classes)
        
        # self.bi_lstm = nn.LSTM(62, 62, num_layers=1,  bidirectional=True)
        # self.up_weight = up_weight

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x5, _ = self.bi_lstm(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNet1d(1, 1)
    print(model)
