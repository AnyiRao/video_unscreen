import cv2
import numpy as np

from ..utils import (dilate_mask, fuse_fgbg, get_fgbox, get_outer_boundary,
                     get_target_size)
from .region_fill import regionfill


class BackgroundAgent():
    """Some naive methods to inpaint the background.

    To make sure that the boundary of the foreground would not be sampled, this
    agent would first dilates the mask, which we name as dilated_mask. Then
    different methods are used to fill this dilated mask, including naive fill
    by mean color, partial convolution and the region fill algorithm.
    References:
    1. partial convolution
    paper: https://arxiv.org/abs/1804.07723
    The orginal partial conv would have a learned weights for the conv kernel
    while here we just need to set the all the weights in the kernel
    as one
    2. region fill
    the algorithm comes from matlab:
    https://www.mathworks.com/help/images/ref/imfill.html
    the python code is modified from:
    https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/tree/master/utils


    Args:
        input_long_side (int): long side of the input image would be resize to
        dilateion_ksize (int): the kernel size for mask dilation
        dilation_iters (int): the iterations for mask dilation
        boundary_ksize (int): the kernel size for getting the boundary of the
            dilated mask, it would be used for getting the mean color
        dilation_iters (int): the iterations for getting the boundary of the
            dilated mask, it would be used for getting the mean color
        pcov_ksize (int): the kernel size of partial convolution,
            only use for partial convolution

    Attributes:
        input_long_side (int): long side of the input image would be resize to
        dilateion_ksize (int): the kernel size for mask dilation
        dilation_iters (int): the iterations for mask dilation
        boundary_ksize (int): the kernel size for getting the boundary of the
            dilated mask, it would be used for getting the mean color
        dilation_iters (int): the iterations for getting the boundary of the
            dilated mask, it would be used for getting the mean color
        pcov_ksize (int): the kernel size of partial convolution,
            only use for partial convolution
    """

    def __init__(self,
                 input_long_side=540,
                 dilation_ksize=5,
                 dilation_iters=3,
                 boundary_ksize=7,
                 boundary_iters=10,
                 pcov_ksize=5):
        self.input_long_side = input_long_side
        self.dilation_ksize = dilation_ksize
        self.dilation_iters = dilation_iters
        self.boundary_ksize = boundary_ksize
        self.boundary_iters = boundary_iters
        self.pcov_ksize = pcov_ksize

    def get_mean_bg(self, img_hsv, mask):
        """model the background as a mean color, the mean color is the mean
        value of the samples in the boundary, which is calculated in the HSV
        color space.

        Args:
            img_hsv (np.array<uint8>): the input image, in HSV color space
            mask (np.array<uint8>): the mask, 255 for foreground and 0 for
                background

        Returns:
            bgimg_hsv (np.array<uint8>): the estimated background image, in HSV
                color space, the shape of which is the same to img_hsv
        """
        boundary = get_outer_boundary(mask, self.boundary_ksize,
                                      self.boundary_iters)
        boundary[boundary > 0] = 1
        boundary = boundary.astype(np.bool)
        bg_pixels = boundary.sum()
        if bg_pixels == 0:
            bg_color = np.mean(img_hsv, axis=(0, 1))
        else:
            bg_color = (img_hsv *
                        boundary[..., np.newaxis]).sum(axis=(0, 1)) / bg_pixels
            bg_color = bg_color.astype(np.uint8)

        bgimg_hsv = np.zeros(img_hsv.shape, np.uint8)
        for i in range(3):
            bgimg_hsv[:, :, i] = bg_color[i]
        return bgimg_hsv

    def get_bg_by_pcov(self, img, mask):
        """inpaint the hole (region covered by foreground mask) with partial
        convolution.

        Args:
            img (np.array<uint8>): the input image, in BGR color space
            mask (np.array<uint8>): the mask, 255 for foreground and 0 for
                background

        Returns:
            bgimg (np.array<uint8>): the estimated background image, in BGR
                color space, the shape of which is the same to img
        """
        bgimg = img.copy()
        bgimg[mask > 0] = 0
        count = (mask == 0).astype(np.float)
        # find the region of interest to apply convolution
        x_min, x_max, y_min, y_max = get_fgbox(mask, padsize=self.pcov_ksize)
        num_pixels = (x_max - x_min) * (y_max - y_min)
        count = count[x_min:x_max, y_min:y_max]
        img_roi = bgimg[x_min:x_max, y_min:y_max]
        MAX_ITERS = 100
        for _ in range(MAX_ITERS):
            img_roi = cv2.boxFilter(img_roi, -1,
                                    (self.pcov_ksize, self.pcov_ksize))
            count = cv2.boxFilter(count, -1,
                                  (self.pcov_ksize, self.pcov_ksize))
            img_roi[count > 0] = np.clip(
                (img_roi[count > 0] / count[count > 0][..., np.newaxis]), 0,
                255)
            count[count > 0] = 1
            if count.sum() >= num_pixels:
                break
        bgimg[x_min:x_max, y_min:y_max] = img_roi
        bgimg = bgimg.astype(np.uint8)
        return bgimg

    def get_bg_by_regionfill(self, img_hsv, mask):
        """inpaint the hole (region covered by foreground mask) with
        regionfill.

        Here we only fill the V channel.
        For the H and S channel, we use the mean value achieved by
        "get_mean_bg"

        Args:
            img_hsv (np.array<uint8>): the input image, in HSV color space
            mask (np.array<uint8>): the mask, 255 for foreground and 0 for
                background

        Returns:
            bgimg_hsv (np.array<uint8>): the estimated background image, in HSV
                color space, the shape of which is the same to img_hsv
        """
        average_bg = self.get_mean_bg(img_hsv, mask)
        bgimg_hsv = img_hsv.copy()
        inpainted_vchannel = regionfill(bgimg_hsv[:, :, -1], mask > 0,
                                        0.5).astype(np.uint8)
        bgimg_hsv[mask > 0] = average_bg[mask > 0]
        bgimg_hsv[:, :, -1][mask > 0] = inpainted_vchannel[mask > 0]
        return bgimg_hsv

    def forward(self, img, mask, method='rf'):
        """main function to inpaint the hole (region covered by foreground
        mask)

        Args:
            img (np.array<uint8>): the input image, in BGR color space
            mask (np.array<uint8>): the mask, 255 for foreground and 0 for
                background
            method (str): different methods for inpainting, includeing
                mean: model the background as the mean color of the boundary
                pcov: fill the hole with partial convolution
                rf: fill the hole with regionfill

        Returns:
            bgimg (np.array<uint8>): the estimated background image, in BGR
                color space, the shape of which is the same to img
        """
        ori_h, ori_w = mask.shape
        # if no background
        if (mask == 0).sum() == 0:
            return np.zeros(img.shape)
        # if no foreground
        if mask.sum() == 0:
            return img
        # resize if need
        input_h, input_w = get_target_size(ori_h, ori_w, self.input_long_side)
        img = cv2.resize(img, (input_w, input_h))
        mask = cv2.resize(mask, (input_w, input_h))
        # dilate the mask, since there may be some noise (pixel from
        # foreground) in the tight mask
        dilated_mask = dilate_mask(mask, self.dilation_ksize,
                                   self.dilation_iters)
        # run inpainting
        if method == 'mean':
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bgimg_hsv = self.get_mean_bg(img_hsv, dilated_mask)
            bgimg = cv2.cvtColor(bgimg_hsv, cv2.COLOR_HSV2BGR)
            bgimg = fuse_fgbg(bgimg, img, dilated_mask)
        elif method == 'pcov':
            bgimg = self.get_bg_by_pcov(img, dilated_mask)
            bgimg = fuse_fgbg(bgimg, img, dilated_mask)
        elif method == 'rf':
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bgimg_hsv = self.get_bg_by_regionfill(img_hsv, dilated_mask)
            bgimg = cv2.cvtColor(bgimg_hsv, cv2.COLOR_HSV2BGR)
        else:
            raise NameError(
                f'No such method for background inpainting: {method}')
        bgimg = cv2.resize(bgimg, (ori_w, ori_h))
        return bgimg
