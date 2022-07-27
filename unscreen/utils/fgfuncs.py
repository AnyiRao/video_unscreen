"""some functions for calculating foreground."""
import cv2
import numpy as np
import torch
import pdb
from .imgprocess import get_target_size


def is_pixel_inrange(img, bgimg, winsize=(20, 20, 120), long_side_input=-1):
    """check whether the pixels are in range (bg-winsize//2, bg+winsize//2)

    To check the pixels of an image is near to the background.
    It can be used to remove some obvious false positive in unscreen removal.

    Args:
        img (np.array<np.uint8>): input image, in BGR color space
        bgimg (np.array<np.uint8>): the background, in BGR color space
            can be (h, w, 3) or (3,)
        winsize (Tuple[int]): the window size of each channel (h, s, v)
        long_side_input (int): the target long side to resize,
            if set as a number <= 0, the image would not be resized

    Returns:
        mask (np.array<bool>): a boolen mask to indicate
            whether each pixel is in range
    """
    assert bgimg.ndim == 3 or bgimg.ndim == 1
    h, w = img.shape[:2]
    if long_side_input <= 0:
        input_h, input_w = h, w
    else:
        input_h, input_w = get_target_size(h, w, long_side_input)
        img = cv2.resize(img, (input_w, input_h))
        if bgimg.ndim == 3:
            bgimg = cv2.resize(bgimg, (input_w, input_h))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # if a background image is given
    if bgimg.ndim == 3:
        bgimg_hsv = cv2.cvtColor(bgimg, cv2.COLOR_BGR2HSV)
        img_hsv_tensor = torch.from_numpy(img_hsv).short()
        bgimg_hsv_tensor = torch.from_numpy(bgimg_hsv).short()
        thr_tensor = torch.ShortTensor(np.array(winsize) // 2)
        # clamp with (10, 255) to exclude black pixels
        lower_thr = torch.clamp(bgimg_hsv_tensor - thr_tensor, 10, 255)
        upper_thr = torch.clamp(bgimg_hsv_tensor + thr_tensor, 10, 255)
        flag1 = torch.ge(img_hsv_tensor, lower_thr).sum(dim=-1)
        flag2 = torch.ge(upper_thr, img_hsv_tensor).sum(dim=-1)
        mask = torch.eq(flag1 + flag2, 6)
        mask = mask.numpy().astype(np.uint8)
        if long_side_input > 0:
            mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
        mask = mask > 0
    # if a background color is given
    else:
        bgimg_hsv = cv2.cvtColor(bgimg[np.newaxis, np.newaxis, ...],
                                 cv2.COLOR_BGR2HSV).squeeze()
        winsize = np.array(winsize) // 2
        lower_thr = np.clip(bgimg_hsv - winsize, 10, 255)
        upper_thr = np.clip(bgimg_hsv + winsize, 10, 255)
        mask = cv2.inRange(img_hsv, lower_thr, upper_thr)
        mask = mask.astype(np.uint8)
        if long_side_input > 0:
            mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
        mask = mask > 0
    return mask


def get_fg_naive(img, alpha):
    """a naive method to get foreground without background.

    Args:
        img (np.array<np.uint8>): the input image, in BGR color space
        alpha (np.array<np.uint8>): the alpha channel of the image

    Returns:
        fg (np.array<np.uint8>): the estimated foreground, in BGR color space
    """
    alpha = alpha.astype(np.float) / 255.
    fg = img.astype(np.float) * alpha[..., np.newaxis]
    fg = fg.astype(np.uint8)
    return fg


def get_fg(img, alpha, bg):
    """to get foreground with background.

    The foreground is calculated accrounding the equation:
        img = alpha * fg + (1-alpha) * bg

    Args:
        img (np.array<np.uint8>): the input image, in BGR color space
        alpha (np.array<np.uint8>): the alpha channel of the image
        bg (np.array<np.uint8>): the background image, in BGR color space

    Returns:
        fg (np.array<np.uint8>): the estimated foreground, in BGR color space;
            note that here the fg is in fact alpha*fg, which would be better
            for visual sense
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    alpha_tensor = torch.from_numpy(alpha)
    alpha_tensor = (alpha_tensor.float() / 255.).unsqueeze(dim=-1)
    bg_tesnor = torch.from_numpy(bg).float()
    img_tensor = torch.from_numpy(img).float()
    fg_tensor = img_tensor - (1 - alpha_tensor) * bg_tesnor
    fg_tensor = torch.clamp(fg_tensor, 0, 255)
    fg = fg_tensor.numpy().astype(np.uint8)
    fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
    return fg


def get_bg(alpha, bg):
    """to get foreground with background

    The foreground is calculated accrounding the equation:
        img = alpha * fg + (1-alpha) * bg

    Args:
        img (np.array<np.uint8>): the input image, in BGR color space
        alpha (np.array<np.uint8>): the alpha channel of the image
        bg (np.array<np.uint8>): the background image, in BGR color space
    
    Returns:
        fg (np.array<np.uint8>): the estimated foreground, in BGR color space;
            note that here the fg is in fact alpha*fg, which would be better
            for visual sense
    """
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    alpha_tensor = torch.from_numpy(alpha)
    alpha_tensor = (alpha_tensor.float() / 255.).unsqueeze(dim=-1)
    bg_tesnor = torch.from_numpy(bg).float()
    bg_tensor = (1 - alpha_tensor) * bg_tesnor
    bg_tensor = torch.clamp(bg_tensor, 0, 255)
    bg = bg_tensor.numpy().astype(np.uint8)
    bg = cv2.cvtColor(bg, cv2.COLOR_HSV2BGR)
    return bg


def get_fg_with_colorremove(img,
                            alpha,
                            bg,
                            winsize=(10, 100, 120),
                            long_side_input=960):
    """to get foreground with background and remove the possible background
    pixels.

    At first the alpah is ensemble with a simple threshold method,
    i.e. "is_pixel_inrange".
    Then the foreground is calculated accrounding the equation:
        img = alpha * fg + (1-alpha) * bg

    Args:
        img (np.array<np.uint8>): the input image, in BGR color space
        alpha (np.array<np.uint8>): the alpha channel of the image
        bg (np.array<np.uint8>): the background image, in BGR color space
        winsize (Tuple[int]): the window size of each channel (h, s, v)
        long_side_input (int): the target long side to resize,
            if set as -1, the image would not be resized

    Returns:
        fg (np.array<np.uint8>): the estimated foreground, in BGR color space;
            note that here the fg is in fact alpha*fg, which would be better
            for visual sense
    """
    bg_mask = is_pixel_inrange(img, bg, winsize, long_side_input)
    alpha[bg_mask] = 0
    fg = get_fg(img, alpha, bg)
    return fg


def composite_fgbg(fg, alpha, bg, extend=False):
    """composite foreground and background to be an image.

    Args:
        fg (np.array<np.uint8>): the foreground image, in BGR color space
        alpha (np.array<np.uint8>): the alpha channel of the image
        bg (np.array<np.uint8>): the background image, in BGR color space
        extend (bool): whether to keep the extended background or not,
            since the h/w ratio of the fg may not match the bg, we would
            resize the background to cover all the foreground.
            Then if extend==True, the image would be returned.
            If extend==False, the area of the foreground would be cropped
            and then returned.

    Returns:
        composition (np.array<np.uint8>): the composition result,
            in BGR color space
    """
    fg_h, fg_w = fg.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    if float(fg_h) / fg_w > float(bg_h) / bg_w:
        new_bg_h = fg_h
        new_bg_w = int(float(bg_w) * new_bg_h / bg_h)
    else:
        new_bg_w = fg_w
        new_bg_h = int(float(bg_h) * new_bg_w / bg_w)
    bg = cv2.resize(bg, (new_bg_w, new_bg_h))
    left = max(new_bg_w // 2 - fg_w // 2, 0)
    right = left + fg_w
    top = max(new_bg_h // 2 - fg_h // 2, 0)
    bottom = top + fg_h
    alpha = alpha.astype(np.float) / 255.
    alpha[alpha > 0.9] = 1  # to avoide fg being blur
    fg = fg.astype(np.float)
    bg_roi = bg[top:bottom, left:right].astype(np.float)
    composition = fg + bg_roi * (1 - alpha[..., np.newaxis])
    composition = composition.clip(0, 255)
    composition = composition.astype(np.uint8)
    if extend:
        composition_entend = bg.copy()
        composition_entend[top:bottom, left:right] = composition
        composition = composition_entend
    return composition
