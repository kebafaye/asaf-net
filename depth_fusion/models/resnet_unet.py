"""ResNet-UNet baseline (compatible with legacy checkpoints)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetUNet(nn.Module):
    """Encoder-decoder fusion network (4 or 5 input channels)."""

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        assert backbone in ("resnet34", "resnet50")
        weights = models.ResNet34_Weights.DEFAULT if backbone == "resnet34" and pretrained else None
        if backbone == "resnet34":
            res = models.resnet34(weights=weights)
            filters = [64, 64, 128, 256, 512]
        else:
            res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            filters = [64, 256, 512, 1024, 2048]

        if in_channels != 3:
            w = res.conv1.weight.data
            new_conv = nn.Conv2d(
                in_channels,
                res.conv1.out_channels,
                kernel_size=res.conv1.kernel_size,
                stride=res.conv1.stride,
                padding=res.conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = w
                if in_channels > 3:
                    mean_w = w.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        new_conv.weight[:, c : c + 1] = mean_w
            res.conv1 = new_conv

        self.enc0 = nn.Sequential(res.conv1, res.bn1, res.relu)
        self.pool0 = res.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = res.layer1, res.layer2, res.layer3, res.layer4

        def conv_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.up4 = conv_block(filters[4] + filters[3], 512)
        self.up3 = conv_block(512 + filters[2], 256)
        self.up2 = conv_block(256 + filters[1], 128)
        self.up1 = conv_block(128 + filters[0], 64)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        p0 = self.pool0(e0)
        e1 = self.enc1(p0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        u4 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.up4(torch.cat([u4, e3], dim=1))
        u3 = F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, e2], dim=1))
        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, e1], dim=1))
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e0], dim=1))
        return self.out_conv(u1)
