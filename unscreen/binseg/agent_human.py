from __future__ import division
import time
import cv2
import numpy as np
import pdb
import os
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .human_parse import network
from ..utils import pad_resize, inv_pad_resize, imnormalize

import warnings
warnings.simplefilter("ignore", UserWarning)

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    }
}

num_classes = dataset_settings['lip']['num_classes']
input_size = dataset_settings['lip']['input_size']
label = dataset_settings['lip']['label']


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def transform_logits(logits, center, scale, width, height, input_size):
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),  # (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits, axis=2)

    return target_logits


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class HumanSegAgent():
    def __init__(self,
                 model_path=None,
                 input_long_side=912,
                 downscale=1, crop_h=473, crop_w=473,
                 stride_ratio=1/2.,
                 flip=True,
                 cuda_device=0):
        self.input_size = input_size
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.input_size = np.asarray(input_size)

        self.model_path = model_path
        self.model = network(num_classes=num_classes, pretrained=None).cuda()
        self.model = nn.DataParallel(self.model)
        self.cuda_device = cuda_device
        # self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # if self.cuda_device >= 0:
        #     self.model.cuda(self.cuda_device)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.division = 1
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.flip = flip
        self.input_long_side = input_long_side
        self.stride_ratio = stride_ratio

    def get_input_size(self, h, w):
        if h > w:
            input_h = self.input_long_side
            input_w = int(float(self.input_long_side) * w / h)
            if input_w % self.division != 0:
                input_w = (input_w // self.division + 1) * self.division
        else:
            input_w = self.input_long_side
            input_h = int(float(self.input_long_side) * h / w)
            if input_h % self.division != 0:
                input_h = (input_h // self.division + 1) * self.division
        if self.crop_h > input_h:
            input_h = self.crop_h
        if self.crop_w > input_w:
            input_w = self.crop_w
        return input_h, input_w

    def grid_crop(self, img, stride_ratio=2/3):
        imglist = []
        locationlist = []
        h, w, _ = img.shape
        stride_h = int(np.ceil(self.crop_h*stride_ratio))
        stride_w = int(np.ceil(self.crop_w*stride_ratio))
        grid_h = int(np.ceil(float(h-self.crop_h)/stride_h) + 1)
        grid_w = int(np.ceil(float(w-self.crop_w)/stride_w) + 1)
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + self.crop_h, h)
                s_h = e_h - self.crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + self.crop_w, w)
                s_w = e_w - self.crop_w
                img_crop = img[s_h:e_h, s_w:e_w].copy()
                imglist.append(img_crop)
                locationlist.append([s_h, e_h, s_w, e_w])
                if self.flip:
                    imglist.append(cv2.flip(img_crop, 1))
                    locationlist.append([s_h, e_h, e_w, s_w])
        return np.array(imglist), np.array(locationlist)

    def inv_grid_crop(self, preds_crop, locations):
        h = locations[:, 1].max()
        w = locations[:, 3].max()
        num_class = preds_crop.shape[-1]
        num_crops = preds_crop.shape[0]
        pred = np.zeros((h, w, num_class), dtype=float)
        count_crop = np.zeros((h, w, 1), dtype=float)
        for i in range(num_crops):
            s_h, e_h, s_w, e_w = locations[i]
            assert s_h <= e_h
            if s_w > e_w:
                pred[s_h:e_h, e_w:s_w] += preds_crop[i, :, ::-1]
                count_crop[s_h:e_h, s_w:e_w] += 1
            else:
                pred[s_h:e_h, s_w:e_w] += preds_crop[i]
                count_crop[s_h:e_h, s_w:e_w] += 1
        pred /= count_crop
        return pred

    def preprocess(self, img):
        h, w, _ = img.shape
        input_size = self.get_input_size(h, w)
        img, _ = pad_resize(img, input_size)
        print(img.shape)
        pdb.set_trace()
        img = imnormalize(img)
        imgs_crop, locations = self.grid_crop(img, self.stride_ratio)
        input_tensor = torch.from_numpy(imgs_crop.transpose(0, 3, 1, 2)).float()
        return input_tensor, locations

    def postprocess(self, output, locations, ori_size):
        """
            locations: (n, 4) location of each cropped patch in the image
            ori_size: (h, w) of the input image (before crop)
        """
        _, _, h_o, w_o = output.size()
        if (h_o != self.crop_h) or (w_o != self.crop_w):
            output = F.interpolate(output, (self.crop_h, self.crop_w), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        output = output.numpy()
        preds_crop = output.transpose(0, 2, 3, 1)

        pred_score = self.inv_grid_crop(preds_crop, locations)
        pred_score = inv_pad_resize(pred_score, ori_size)
        pred = np.argmax(pred_score, axis=2)
        pred = (pred*255).astype(np.uint8)
        return pred

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def forward(self, img):
        """main function of the SegAgent, given an image, it would output a mask
            to indicate whether it is foreground (255) or background (0)

        Args:
            img (np.array<uint8>): shape (h, w, 3),
                the input img, in BGR color space

        Returns:
            pred (np.array<uint8>): shape (h, w),
                the final predicted mask, 255 as foreground and 0 as background
        """
        # cv2.imwrite("test1.jpg", img)
        # input_tensor, locations = self.preprocess(img)

        h, w, _ = img.shape
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        c = person_center

        # cv2.imwrite("test2.jpg", img)

        img = imnormalize(img)
        img = img[np.newaxis, :]
        input_tensor = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().cuda()
        with torch.no_grad():
            output = self.model(input_tensor)
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output)
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
        parsing_result = np.argmax(logits_result, axis=2)
        parsing_result[parsing_result > 0] = 255

        logits_result_norm = softmax(logits_result, axis=2)
        pred_score = np.stack((logits_result_norm[:, :, 0], np.sum(logits_result_norm[:, :, 1:], axis=2)), axis=2)

        pred = np.asarray(parsing_result, dtype=np.uint8)
        # return pred, pred_score
        return pred
