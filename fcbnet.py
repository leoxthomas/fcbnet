"""
========
PyTorch implementation of FCBNet for semantic segmentation.
========

Reference
---------
Ramos et al., "FCBNet: A Parameter-efficient Convolutional Approach for
Camouflaged Weed Detection in Multispectral Aerial Imagery", CVPR Workshops 2026.
"""


from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class FeatureCorrectionBlock(nn.Module):


    def __init__(
        self,
        channels: int,
        bottleneck_ratio: int = 4,
        gn_groups: int = 8,
        alpha_init: float = 0.0,
        alpha_per_channel: bool = False,
        dw_kernel_size: int = 3,
        min_hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        if bottleneck_ratio < 1:
            raise ValueError(f"bottleneck_ratio must be >= 1, got {bottleneck_ratio}")


        hidden = max(min_hidden_channels, channels // bottleneck_ratio)


        def _pick_gn_groups(num_channels: int, preferred: int) -> int:
            g = min(preferred, num_channels)
            while g > 1 and (num_channels % g) != 0:
                g -= 1
            return g

        gn1 = _pick_gn_groups(hidden, gn_groups)
        gn2 = _pick_gn_groups(hidden, gn_groups)

        padding = dw_kernel_size // 2

        self.down = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=gn1, num_channels=hidden, eps=1e-6, affine=True)
        self.act1 = nn.GELU()

        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=dw_kernel_size,
            stride=1,
            padding=padding,
            groups=hidden,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=gn2, num_channels=hidden, eps=1e-6, affine=True)
        self.act2 = nn.GELU()

        self.up = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)

        if alpha_per_channel:
            self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), float(alpha_init)))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = self.gn1(y)
        y = self.act1(y)

        y = self.dw(y)
        y = self.gn2(y)
        y = self.act2(y)

        y = self.up(y)
        return x + self.alpha * y


class ConvNeXtEncoderWithFCB(nn.Module):


    def __init__(
        self,
        variant: str = "base",
        in_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        fcb_bottleneck_ratio: int = 4,
        fcb_gn_groups: int = 8,
        fcb_alpha_init: float = 0.0,
        fcb_alpha_per_channel: bool = False,
        fcb_dw_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.variant = variant
        self.in_channels = int(in_channels)

        backbone = self._build_convnext(variant=variant, pretrained=pretrained)
        self.backbone = backbone

        self._replace_stem_conv_in_channels(self.in_channels)
        if freeze_backbone:
            self._freeze_backbone()

        self.stem = backbone.features[0]
        self.stage1 = backbone.features[1]
        self.down1 = backbone.features[2]
        self.stage2 = backbone.features[3]
        self.down2 = backbone.features[4]
        self.stage3 = backbone.features[5]
        self.down3 = backbone.features[6]
        self.stage4 = backbone.features[7]

        c1, c2, c3, c4 = self._infer_stage_channels()

        self.stage_channels = (c1, c2, c3, c4)

        self.fcb1 = FeatureCorrectionBlock(
            channels=c1,
            bottleneck_ratio=fcb_bottleneck_ratio,
            gn_groups=fcb_gn_groups,
            alpha_init=fcb_alpha_init,
            alpha_per_channel=fcb_alpha_per_channel,
            dw_kernel_size=fcb_dw_kernel_size,
        )
        self.fcb2 = FeatureCorrectionBlock(
            channels=c2,
            bottleneck_ratio=fcb_bottleneck_ratio,
            gn_groups=fcb_gn_groups,
            alpha_init=fcb_alpha_init,
            alpha_per_channel=fcb_alpha_per_channel,
            dw_kernel_size=fcb_dw_kernel_size,
        )
        self.fcb3 = FeatureCorrectionBlock(
            channels=c3,
            bottleneck_ratio=fcb_bottleneck_ratio,
            gn_groups=fcb_gn_groups,
            alpha_init=fcb_alpha_init,
            alpha_per_channel=fcb_alpha_per_channel,
            dw_kernel_size=fcb_dw_kernel_size,
        )
        self.fcb4 = FeatureCorrectionBlock(
            channels=c4,
            bottleneck_ratio=fcb_bottleneck_ratio,
            gn_groups=fcb_gn_groups,
            alpha_init=fcb_alpha_init,
            alpha_per_channel=fcb_alpha_per_channel,
            dw_kernel_size=fcb_dw_kernel_size,
        )

    @staticmethod
    def _build_convnext(variant: str, pretrained: bool) -> nn.Module:
        variant = variant.lower().strip()
        if variant not in {"tiny", "small", "base", "large"}:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")

        fn = getattr(models, f"convnext_{variant}")

        if not pretrained:
            return fn(weights=None)

        weights_enum_name = f"ConvNeXt_{variant.capitalize()}_Weights"
        weights_enum = getattr(models, weights_enum_name, None)
        if weights_enum is None:

            return fn(pretrained=True)

        return fn(weights=weights_enum.DEFAULT)

    def _infer_stage_channels(self) -> Tuple[int, int, int, int]:
        stem_conv: nn.Conv2d = self.backbone.features[0][0]
        c1 = int(stem_conv.out_channels)

        down1_conv: nn.Conv2d = self.backbone.features[2][1]
        down2_conv: nn.Conv2d = self.backbone.features[4][1]
        down3_conv: nn.Conv2d = self.backbone.features[6][1]
        c2 = int(down1_conv.out_channels)
        c3 = int(down2_conv.out_channels)
        c4 = int(down3_conv.out_channels)
        return c1, c2, c3, c4

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _replace_stem_conv_in_channels(self, in_channels: int) -> None:
        stem = self.backbone.features[0]
        old_conv: nn.Conv2d = stem[0]
        if old_conv.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                w_old = old_conv.weight
                w_new = new_conv.weight
                mean_rgb = w_old.mean(dim=1, keepdim=True)
                w_new.copy_(mean_rgb.repeat(1, in_channels, 1, 1))

                if new_conv.bias is not None and old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

        stem[0] = new_conv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feats: Dict[str, torch.Tensor] = {}

        x = self.stem(x)

        x = self.stage1(x)
        x = self.fcb1(x)
        feats["stage1"] = x

        x = self.down1(x)
        x = self.stage2(x)
        x = self.fcb2(x)
        feats["stage2"] = x

        x = self.down2(x)
        x = self.stage3(x)
        x = self.fcb3(x)
        feats["stage3"] = x

        x = self.down3(x)
        x = self.stage4(x)
        x = self.fcb4(x)
        feats["stage4"] = x

        return x, feats


class FPNDecoder(nn.Module):


    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        fpn_dim: int = 128,
        refine_depth: int = 2,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(f"in_channels must be a 4-tuple (C1,C2,C3,C4), got {in_channels}")

        c1, c2, c3, c4 = [int(x) for x in in_channels]
        self.in_channels = (c1, c2, c3, c4)
        self.fpn_dim = int(fpn_dim)
        self.refine_depth = int(refine_depth)


        def _pick_gn_groups(num_channels: int, preferred: int) -> int:
            g = min(preferred, num_channels)
            while g > 1 and (num_channels % g) != 0:
                g -= 1
            return g

        gn = _pick_gn_groups(self.fpn_dim, gn_groups)

        def smooth_block() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.fpn_dim),
                nn.GELU(),
            )


        self.l1 = nn.Conv2d(c1, self.fpn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2 = nn.Conv2d(c2, self.fpn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.l3 = nn.Conv2d(c3, self.fpn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.l4 = nn.Conv2d(c4, self.fpn_dim, kernel_size=1, stride=1, padding=0, bias=False)


        self.s4 = smooth_block()
        self.s3 = smooth_block()
        self.s2 = smooth_block()
        self.s1 = smooth_block()


        self.refine = nn.Sequential(*[smooth_block() for _ in range(self.refine_depth)])

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        s1 = feats["stage1"]
        s2 = feats["stage2"]
        s3 = feats["stage3"]
        s4 = feats["stage4"]


        l1 = self.l1(s1)
        l2 = self.l2(s2)
        l3 = self.l3(s3)
        l4 = self.l4(s4)


        p4 = self.s4(l4)

        p3 = self._upsample_like(p4, l3)
        p3 = self.s3(l3 + p3)

        p2 = self._upsample_like(p3, l2)
        p2 = self.s2(l2 + p2)

        p1 = self._upsample_like(p2, l1)
        p1 = self.s1(l1 + p1)


        if self.refine_depth > 0:
            p1 = self.refine(p1)

        return p1


class FCBNet(nn.Module):

    def __init__(
        self,
        *,
        variant: str = "base",
        in_channels: int = 3,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = True,

        fcb_bottleneck_ratio: int = 2,
        fcb_gn_groups: int = 8,
        fcb_alpha_init: float = 0.07,
        fcb_alpha_per_channel: bool = False,
        fcb_dw_kernel_size: int = 3,

        fpn_dim: int = 128,
        refine_depth: int = 2,
        decoder_gn_groups: int = 8,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "For CrossEntropyLoss, num_classes should be >= 2. "
                f"Got num_classes={num_classes}."
            )

        self.num_classes = int(num_classes)

        self.encoder = ConvNeXtEncoderWithFCB(
            variant=variant,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            fcb_bottleneck_ratio=fcb_bottleneck_ratio,
            fcb_gn_groups=fcb_gn_groups,
            fcb_alpha_init=fcb_alpha_init,
            fcb_alpha_per_channel=fcb_alpha_per_channel,
            fcb_dw_kernel_size=fcb_dw_kernel_size,
        )


        if not hasattr(self.encoder, "stage_channels"):
            raise RuntimeError("encoder.stage_channels is missing. Add it in ConvNeXtEncoderWithFCB.__init__().")
        stage_channels = tuple(self.encoder.stage_channels)

        self.decoder = FPNDecoder(
            in_channels=stage_channels,
            fpn_dim=fpn_dim,
            refine_depth=refine_depth,
            gn_groups=decoder_gn_groups,
        )


        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(fpn_dim, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        h, w = x.shape[-2], x.shape[-1]

        x_final, feats = self.encoder(x)
        dec = self.decoder(feats)
        logits = self.head(dec)

        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits
