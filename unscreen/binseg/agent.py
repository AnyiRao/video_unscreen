import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pdb
from ..utils import imnormalize, inv_pad_resize, pad_resize
from .deeplab_model import get_deeplab_model

warnings.simplefilter('ignore', UserWarning)


class SegAgent():
    """Binary Segmentation, to classify each pixel in an image as foreground or
    backgournd.

    Args:
        model_path (str): the path of the weights of the deep model
        input_long_side (int): long side of the input image would be resize to
        crop_h (int): the height of the cropped patch, the input image would be
            cropped as multiple patches with size of (crop_h, crop_w) and then
            fed to the model
        crop_w (int): the width of the cropped patch
        stride_ratio (float): the stride for cropping would be
            stride_ratio*crop_h or stride*crop_w
        flip (bool): whether to flip the image and ensemble with the flip
            result
        cuda_device (int): the device to run the model,
            if set as negative, it would run in CPU

    Attributes:
        model (nn.Module): the deep model, only support deeplabv3 plus now
        cuda_device (int): the device to run the model
        input_long_side (int): long side of the input image would be resize to
        crop_h (int): the height of the cropped patch, the input image would be
            cropped as multiple patches with size of (crop_h, crop_w) and then
            fed to the model
        crop_w (int): the width of the cropped patch
        stride_ratio (float): the stride for cropping would be
            stride_ratio*crop_h or stride*crop_w
        flip (bool): whether to flip the image and ensemble with the flip
            result
        division (int): the division that need to meet when resize and pad the
            image here we set as 1 means that it can be resize to any number
    """

    def __init__(self,
                 model_path=None,
                 input_long_side=912,
                 crop_h=513,
                 crop_w=513,
                 stride_ratio=1 / 2.,
                 flip=True,
                 cuda_device=0):
        self.model = get_deeplab_model()
        self.cuda_device = cuda_device
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        if self.cuda_device >= 0:
            self.model.cuda(self.cuda_device)
        self.model.eval()

        self.division = 1
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.flip = flip
        self.input_long_side = input_long_side
        self.stride_ratio = stride_ratio

    def get_target_size(self, h, w):
        """calculate the target size of the image according to the original
        size and self.input_long_side.

        The long side of the image, i.e. max(h, w), is resized to
        self.input_long_side, and the short side is resized accroding to the
        ratio, i.e. self.input_long_side/long_side.

        Args:
            h (int): original height of the input image
            w (int): original width of the input image

        Returns:
            target_h (int): the resized height
            target_w (int): the resized width
        """
        if h > w:
            target_h = self.input_long_side
            target_w = int(float(self.input_long_side) * w / h)
            if target_w % self.division != 0:
                target_w = (target_w // self.division + 1) * self.division
        else:
            target_w = self.input_long_side
            target_h = int(float(self.input_long_side) * h / w)
            if target_h % self.division != 0:
                target_h = (target_h // self.division + 1) * self.division
        if self.crop_h > target_h:
            target_h = self.crop_h
        if self.crop_w > target_w:
            target_w = self.crop_w
        return target_h, target_w

    def grid_crop(self, img, stride_ratio):
        """grid crop the image into multiple patches.

        It is similar to sliding window with a step of stride_ratio*crop_h and
        stride_ratio*crop_h.

        Args:
            img (np.array<uint8>): the input image to be cropped
            stride_ratio (float):  the stride ratio for cropping

        Returns:
            imglist (np.array<uint8>): shape (N, img.shape),
                the cropped N patches
            locationlist (np.array<int>): shape (N, 4),
                the location, i.e (top, bottom, left, right), of each patch
                in the image. If flip, the location would be
                (top, bottom, right, left).
        """
        imglist = []
        locationlist = []
        h, w, _ = img.shape
        stride_h = int(np.ceil(self.crop_h * stride_ratio))
        stride_w = int(np.ceil(self.crop_w * stride_ratio))
        grid_h = int(np.ceil(float(h - self.crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(w - self.crop_w) / stride_w) + 1)
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
        """ensemble the predictions of the cropped patches.

        Args:
            preds_crop (np.array<float>): shape (N, crop_h, crop_w, numclass),
                the predictions of the patches
            locationlist (np.array<int>): shape (N, 4),
                the location, i.e (top, bottom, left, right), of each patch
                in the image. If a patch is cropped from a flip image,
                the location would be (top, bottom, right, left)

        Returns:
            pred (np.array<float>): shape (h, w, numclass)),
                the ensembled prediction
        """
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
                count_crop[s_h:e_h, e_w:s_w] += 1
            else:
                pred[s_h:e_h, s_w:e_w] += preds_crop[i]
                count_crop[s_h:e_h, s_w:e_w] += 1
        pred /= count_crop
        return pred

    def preprocess(self, img):
        """preprocessing, including resize, pad, normalize, grid_crop and
        to_tensor.

        Args:
            img (np.array<uint8>): the input img, in BGR color space

        Retures:
            input_tensor (torch.Tesnor<float>): shape (N, 3, crop_h, crop_w)
                tensor of the cropped patches to feed to the deep model
            locations (np.array<int>): shape (N, 4),
                the location, i.e (top, bottom, left, right), of each patch
                in the image. If flip, the location would be
                (top, bottom, right, left).
        """
        h, w, _ = img.shape
        input_size = self.get_target_size(h, w)
        img, _ = pad_resize(img, input_size)
        img = imnormalize(img)
        imgs_crop, locations = self.grid_crop(img, self.stride_ratio)
        input_tensor = torch.from_numpy(imgs_crop.transpose(0, 3, 1,
                                                            2)).float()
        return input_tensor, locations

    def postprocess(self, output, locations, ori_size):
        """postprocessing to get the final prediction.

        Postprocessing do the following things:
        1. interpolation if the shape mismatch
        2. softmax
        3. ensemble the cropped predictions
        4. inverse pad and resise
        5. transform the predicted score to label with value {0, 255}

        Args:
            output (torch.Tesnor<float>): (N, numclass, crop_h, crop_w)
            locations (np.array<int>): shape (N, 4),
                the location, i.e (top, bottom, left, right), of each patch
                in the image. If flip, the location would be
                (top, bottom, right, left).
            ori_size: original size, i.e. (h, w), of the input image
                (before cropped)

        Returns:
            pred (np.array<uint8>): shape (h, w),
                the final predicted mask, 255 as foreground and 0 as background
        """
        _, _, h_o, w_o = output.size()
        if (h_o != self.crop_h) or (w_o != self.crop_w):
            output = F.interpolate(
                output, (self.crop_h, self.crop_w),
                mode='bilinear',
                align_corners=True)
        output = F.softmax(output, dim=1)
        output = output.numpy()
        preds_crop = output.transpose(0, 2, 3, 1)

        pred_score = self.inv_grid_crop(preds_crop, locations)
        pred_score = inv_pad_resize(pred_score, ori_size)
        pred = np.argmax(pred_score, axis=2)
        pred = (pred * 255).astype(np.uint8)
        return pred, pred_score

    def forward(self, img):
        """main function of the SegAgent, given an image, it would output a
        mask to indicate whether it is foreground (255) or background (0)

        Args:
            img (np.array<uint8>): shape (h, w, 3),
                the input img, in BGR color space

        Returns:
            pred (np.array<uint8>): shape (h, w),
                the final predicted mask, 255 as foreground and 0 as background
        """
        ori_size = img.shape[:2]
        input_tensor, locations = self.preprocess(img)
        if self.cuda_device >= 0:
            input_tensor = input_tensor.cuda(self.cuda_device)
        with torch.no_grad():
            output = self.model(input_tensor)
        output = output.detach().cpu()
        pred, pred_score = self.postprocess(output, locations, ori_size)
        # return pred, pred_score
        return pred
