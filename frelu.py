import torch.nn as nn
import torch
class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """
    def __init__(self, in_channels):
        super(FReLU,self).__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x= torch.max(x, x1)
        return x


########################################
定义时Bottleneck_FReLU
self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
self.bn1 = norm_layer(group_width)
self.frelu1 = FReLU(group_width)


前向中
out = self.conv1(x)
out = self.bn1(out)
out = self.frelu1(out)   #替换掉Relu
