# general libs
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResBlock(nn.Module):

    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(
                indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv1 = nn.Conv2d(
            indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + r


class Encoder_M(nn.Module):

    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

    def forward(self, in_f, in_m, in_o):
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float()  # add channel dim
        x = self.conv1(in_f) + self.conv1_m(m) + self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1


class Encoder_Q(nn.Module):

    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

    def forward(self, in_f):
        x = self.conv1(in_f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1


class Refine(nn.Module):

    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(
            inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(
            pm,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):

    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(
            1024, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(
            mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p = F.interpolate(
            p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p


class Memory(nn.Module):

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)  # b, thw, c

        qi = q_in.view(B, D_e, H * W)  # b, c, hw

        p = torch.bmm(mi, qi)  # b, thw, hw
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, thw, HW

        mo = m_out.view(B, D_o, T * H * W)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(
            indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(
            indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class STM(nn.Module):

    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)

    def memorize(self, frame, masks):
        """
            frame: (1, 3, h, w)
            masks: (1, 2, h, w)
        """
        # memorize a frame
        r4, _, _, _ = self.Encoder_M(frame, masks[:, 1], masks[:, 0])
        k4, v4 = self.KV_M_r4(r4)  # 1, 128 and 512, h/16, w/16
        return k4, v4

    def soft_aggregation(self, ps):
        _, H, W = ps.shape
        em = torch.zeros(1, 2, H, W).cuda(ps.device)
        em[0, 0] = torch.prod(1 - ps, dim=0)  # bg prob
        em[0, 1] = ps  # obj prob
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def segment(self, frame, keys, values):
        """
            frame: (1, 3, h, w)
            keys: (1, 128, memnum, h//16, 2//16)
            values: (1, 512, memnum, h//16, 2//16)
        """
        # pad
        r4, r3, r2, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)  # 1, dim, h/16, w/16

        m4, _ = self.Memory(keys, values, k4, v4)
        logits = self.Decoder(m4, r3, r2)

        ps = F.softmax(logits, dim=1)[:, 1]  # 1, h, w
        logit = self.soft_aggregation(ps)  # 1, 2, h, w
        return logit

    def forward(self, *args, **kwargs):
        if len(args) > 2:
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)
