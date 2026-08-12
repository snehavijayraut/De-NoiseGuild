"""
model.py
NAFNetSR: Lightweight NAFNet-based super-resolution / denoising architecture
for SEM (Scanning Electron Microscope) grayscale imagery.

Architecture family: NAFNet (Simple Baselines for Image Restoration,
Chen et al. 2022), adapted here as a U-Net style encoder / bottleneck /
decoder with a PixelShuffle reconstruction head for 2x super-resolution
and a bilinear-upsampled residual shortcut to suppress line/space
boundary haloing and ringing artifacts common in SEM metrology imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm operating over the channel dimension of a
    (N, C, H, W) tensor, as used throughout NAFNet."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Splits channels in half along dim=1 and multiplies the two halves
    together. Replaces conventional non-linear activations (GELU/ReLU) as
    the non-linearity in NAFNet, hence "Non-Linear Activation Free"."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Non-Linear Activation Free Block.

    Spatial branch: 1x1 conv -> 3x3 depthwise conv -> SimpleGate ->
    simplified channel attention (SCA) -> 1x1 conv.
    Channel-mixing branch: 1x1 conv -> SimpleGate -> 1x1 conv.
    Both branches are LayerNorm2d pre-normalized and combined with the
    block input via learnable per-channel residual scales (beta, gamma).
    """

    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, kernel_size=3, stride=1, padding=1,
            groups=dw_channel, bias=True,
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, stride=1, padding=0, bias=True)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, stride=1, padding=0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class _Down(nn.Module):
    """Strided-convolution downsampling: halves spatial resolution and
    doubles channel width."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.op(x)


class _Up(nn.Module):
    """Transposed-convolution upsampling: doubles spatial resolution and
    halves channel width."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.op(x)


class NAFNetSR(nn.Module):
    """Lightweight NAFNet-based restoration + super-resolution network.

    Encoder depths (1, 2, 2, 4) at channel widths (32, 64, 128, 256):
      intro conv (in_ch -> dim)
      stage 1: depth[0] NAFBlocks @ dim         -> strided-conv downsample
      stage 2: depth[1] NAFBlocks @ dim*2       -> strided-conv downsample
      stage 3: depth[2] NAFBlocks @ dim*4       -> strided-conv downsample
      bottleneck: depth[3]=4 NAFBlocks @ dim*8
      decoder mirrors the encoder with ConvTranspose2d upsampling and
      additive skip connections.
      reconstruction head: conv -> PixelShuffle(scale) -> conv (dim -> out_ch)
      output = reconstruction(features) + bilinear_upsample(input, scale)
    """

    def __init__(self, in_ch=1, out_ch=1, dim=32, enc_depths=(1, 2, 2, 4), scale=2):
        super().__init__()
        assert len(enc_depths) == 4, "enc_depths must specify exactly 4 stage depths"
        d1, d2, d3, d4 = enc_depths
        c1, c2, c3, c4 = dim, dim * 2, dim * 4, dim * 8
        self.scale = scale

        self.intro = nn.Conv2d(in_ch, c1, kernel_size=3, stride=1, padding=1)

        # Encoder
        self.enc1 = nn.Sequential(*[NAFBlock(c1) for _ in range(d1)])
        self.down1 = _Down(c1, c2)

        self.enc2 = nn.Sequential(*[NAFBlock(c2) for _ in range(d2)])
        self.down2 = _Down(c2, c3)

        self.enc3 = nn.Sequential(*[NAFBlock(c3) for _ in range(d3)])
        self.down3 = _Down(c3, c4)

        # Bottleneck (4-block NAFBlock stack at the deepest 256-channel width)
        self.bottleneck = nn.Sequential(*[NAFBlock(c4) for _ in range(d4)])

        # Decoder (mirrors encoder, with skip-connections)
        self.up3 = _Up(c4, c3)
        self.dec3 = nn.Sequential(*[NAFBlock(c3) for _ in range(d3)])

        self.up2 = _Up(c3, c2)
        self.dec2 = nn.Sequential(*[NAFBlock(c2) for _ in range(d2)])

        self.up1 = _Up(c2, c1)
        self.dec1 = nn.Sequential(*[NAFBlock(c1) for _ in range(d1)])

        # Reconstruction head: PixelShuffle upsampler for `scale`x SR
        self.pre_upsample = nn.Conv2d(c1, c1 * (scale ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=3, stride=1, padding=1)

        # 3 downsampling stages => spatial dims must be divisible by 2^3 = 8
        self.padder_size = 8

    def _check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h or mod_pad_w:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")
        return x

    def forward(self, x):
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        x_pad = self._check_image_size(x)

        feat = self.intro(x_pad)

        e1 = self.enc1(feat)
        d = self.down1(e1)

        e2 = self.enc2(d)
        d = self.down2(e2)

        e3 = self.enc3(d)
        d = self.down3(e3)

        b = self.bottleneck(d)

        u = self.up3(b) + e3
        u = self.dec3(u)

        u = self.up2(u) + e2
        u = self.dec2(u)

        u = self.up1(u) + e1
        u = self.dec1(u)

        u = self.pre_upsample(u)
        u = self.pixel_shuffle(u)
        out = self.out_conv(u)

        # Residual shortcut: bilinear-upsampled copy of the (padded) input,
        # added directly to the reconstructed output. This enforces
        # shortcut residual learning and eliminates line/space boundary
        # haloing / ringing artifacts characteristic of SEM imagery.
        shortcut = F.interpolate(
            x_pad, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        out = out + shortcut

        # Crop back to the exact target (input_size * scale) region,
        # discarding any reflect-padding added for divisibility.
        target_h, target_w = orig_h * self.scale, orig_w * self.scale
        out = out[:, :, :target_h, :target_w]
        return out


if __name__ == "__main__":
    # Quick shape sanity check (run directly: `python model.py`)
    model = NAFNetSR()
    dummy = torch.randn(2, 1, 130, 126)  # deliberately non-multiple-of-8
    y = model(dummy)
    print("input:", tuple(dummy.shape), "output:", tuple(y.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")
