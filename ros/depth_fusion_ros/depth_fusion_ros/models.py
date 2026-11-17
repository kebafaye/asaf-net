import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetUNet(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, in_channels=4, out_channels=1):
        super().__init__()
        assert backbone in ["resnet34", "resnet50"]
        if backbone == "resnet34":
            res = models.resnet34(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        else:
            res = models.resnet50(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]

        if in_channels != 3:
            w = res.conv1.weight  # (64,3,7,7)
            new_conv = nn.Conv2d(
                in_channels,
                res.conv1.out_channels,
                kernel_size=res.conv1.kernel_size,
                stride=res.conv1.stride,
                padding=res.conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                if in_channels > 3:
                    new_conv.weight[:, :3, :, :] = w
                    mean_w = w.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        new_conv.weight[:, c : c + 1, :, :] = mean_w
            res.conv1 = new_conv

        self.enc0 = nn.Sequential(res.conv1, res.bn1, res.relu)
        self.pool0 = res.maxpool
        self.enc1 = res.layer1
        self.enc2 = res.layer2
        self.enc3 = res.layer3
        self.enc4 = res.layer4

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.up4 = conv_block(filters[4] + filters[3], 512)
        self.up3 = conv_block(512 + filters[2], 256)
        self.up2 = conv_block(256 + filters[1], 128)
        self.up1 = conv_block(128 + filters[0], 64)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        p0 = self.pool0(e0)
        e1 = self.enc1(p0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        u4 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = torch.cat([u4, e3], dim=1)
        u4 = self.up4(u4)
        u3 = F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = torch.cat([u3, e2], dim=1)
        u3 = self.up3(u3)
        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, e1], dim=1)
        u2 = self.up2(u2)
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, e0], dim=1)
        u1 = self.up1(u1)

        out = self.out_conv(u1)
        return out
