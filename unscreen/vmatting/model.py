import torch
import torch.nn as nn
from torch.nn import Parameter

# ************** Conv Ops **************


def conv5x5(in_planes, out_planes, stride=1, groups=1, padding=2):
    """5x5 convolution without dilation."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution without dilation."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ************** SpectralNorm **************


def l2_normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """Based on https://github.com/heykeetae/Self-Attention-
    GAN/blob/master/spectral.py  # noqa and add _noupdate_u_v() for
    evaluation."""

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2_normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2_normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            _ = getattr(self.module, self.name + '_u')
            _ = getattr(self.module, self.name + '_v')
            _ = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2_normalize(u.data)
        v.data = l2_normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


# ************** ResNet **************


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = SpectralNorm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SpectralNorm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet backbone for image matting.

    Implement and pre-train on ImageNet with the tricks from
    https://arxiv.org/abs/1812.01187
    without the mix-up part.

    Args:
        # TODO
    """

    def __init__(self,
                 block,
                 layers,
                 trimap_channels,
                 norm_layer=None,
                 late_downsample=False):
        super(ResNet, self).__init__()
        if block == 'BasicBlock':
            block = BasicBlock
        else:
            raise NotImplementedError(f'{block} is not implemented.')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32
        self.start_stride = [1, 2, 1, 2] if late_downsample else [2, 1, 2, 1]
        self.conv1 = SpectralNorm(
            conv3x3(3 + trimap_channels, 32, stride=self.start_stride[0]))
        self.conv2 = SpectralNorm(
            conv3x3(32, self.midplanes, stride=self.start_stride[1]))
        self.conv3 = SpectralNorm(
            conv3x3(
                self.midplanes, self.inplanes, stride=self.start_stride[2]))
        self.bn1 = norm_layer(32)
        self.bn2 = norm_layer(self.midplanes)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=self.start_stride[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.out_channels = 512

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight_bar'):
                    nn.init.xavier_uniform_(m.weight_bar)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        self.conv1.module.weight_bar.data[:, 3:, :, :] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride),
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x1 = self.relu(x)
        x = self.conv3(x1)
        x = self.bn3(x)
        x2 = self.relu(x)

        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x = self.layer4(x5)

        return x, (x1, x2, x3, x4, x5)


# ************** ResNet Decoder**************


class BasicBlock_Dec(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 upsample=None,
                 norm_layer=None,
                 large_kernel=False):
        super(BasicBlock_Dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        conv = conv5x5 if large_kernel else conv3x3
        # Both self.conv1 and self.upsample layers upsample the input
        # when stride != 1
        if self.stride > 1:
            self.conv1 = SpectralNorm(
                nn.ConvTranspose2d(
                    inplanes,
                    inplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False))
        else:
            self.conv1 = SpectralNorm(conv(inplanes, inplanes))
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = SpectralNorm(conv(inplanes, planes))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet_Dec(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 large_kernel=False,
                 norm_layer=None,
                 late_downsample=False):
        super(ResNet_Dec, self).__init__()
        if block == 'BasicBlock_Dec':
            block = BasicBlock_Dec
        else:
            raise NotImplementedError(f'{block} is not implemented.')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        kernel_size = 5 if self.large_kernel else 3

        self.inplanes = in_channels
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = SpectralNorm(
            nn.ConvTranspose2d(
                self.midplanes,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            32, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, self.midplanes, layers[3], stride=2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight_bar'):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock_Dec):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, upsample, norm_layer,
                  self.large_kernel)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    large_kernel=self.large_kernel))

        return nn.Sequential(*layers)

    def forward(self, x, mid_fea):
        x = self.layer1(x)  # (N, 256, 32, 32)
        x = self.layer2(x)  # (N, 128, 64, 64)
        x = self.layer3(x)  # (N, 64, 128, 128)
        x = self.layer4(x)  # (N, 32, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)

        return x


# ************** ResShortCut **************


class ResShortCut(ResNet):

    def __init__(self,
                 block,
                 layers,
                 trimap_channels,
                 norm_layer=None,
                 late_downsample=False):
        super(ResShortCut, self).__init__(block, layers, trimap_channels,
                                          norm_layer, late_downsample)
        first_inplane = 3 + trimap_channels
        self.shortcut_inplane = [first_inplane, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(
                self._make_shortcut(inplane, self.shortcut_plane[stage]))

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(conv3x3(inplane, planes)), nn.ReLU(inplace=True),
            self._norm_layer(planes), SpectralNorm(conv3x3(planes, planes)),
            nn.ReLU(inplace=True), self._norm_layer(planes))

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.relu(out)  # (N, 32, 256, 256)
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.relu(out)

        x2 = self.layer1(out)  # (N, 64, 128, 128)
        x3 = self.layer2(x2)  # (N, 128, 64, 64)
        x4 = self.layer3(x3)  # (N, 256, 32, 32)
        out = self.layer4(x4)  # (N, 512, 16, 16)

        fea1 = self.shortcut[0](x)
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {
            'shortcut': (fea1, fea2, fea3, fea4, fea5),
            'image': x[:, :3, ...]
        }


# ************** ResShortCut Decoder **************


class ResShortCut_Dec(ResNet_Dec):

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 large_kernel=False,
                 norm_layer=None,
                 late_downsample=False):
        super(ResShortCut_Dec,
              self).__init__(block, layers, in_channels, large_kernel,
                             norm_layer, late_downsample)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x = self.layer3(x) + fea3
        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)
        return x


# ************** AudoEncoder **************


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = ResShortCut(
            block='BasicBlock', layers=[2, 2, 2, 2], trimap_channels=4)
        self.decoder = ResShortCut_Dec(
            block='BasicBlock_Dec',
            layers=[2, 2, 2, 2],
            in_channels=self.encoder.out_channels)

    def init_weights(self, pretrained=None):
        self.encoder.init_weights(pretrained)
        self.decoder.init_weights()

    def forward(self, x):
        out, mid_fea = self.encoder(x)
        out = self.decoder(out, mid_fea)
        return out


# ************** Matting Model **************


class UNet(nn.Module):

    def __init__(self, pretrained=None):
        super(UNet, self).__init__()

        self.backbone = SimpleAutoencoder()
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)

    def forward(self, img, alpha_pre, trimap):
        x = torch.cat((img, alpha_pre, trimap), 1)
        raw_alpha = self.backbone(x)
        pred_alpha = (raw_alpha.tanh() + 1.0) / 2.0

        return pred_alpha
