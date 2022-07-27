"""some functions for image processing."""
import cv2
import numpy as np
import torch


def get_center(img):
    '''get the center of a mask region
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    M = cv2.moments(img)
    cX = int(M["m10"] / (M["m00"]+1e-6))
    cY = int(M["m01"] / (M["m00"]+1e-6))
    return (cX, cY)


def get_mask(img):
    '''get masks of a color image.

    Args:
        img: dtype with shape (h, w, 3).
    Returns:
        mask: dtype with shape (h, w, 1) ranging from 0 to 255.
        binary mask: dtype with shape (h, w, 1) ranging from 0 to 1.
    '''
    assert img.shape[2] == 3
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    binary_mask = thresh/255
    return thresh[..., np.newaxis], binary_mask[..., np.newaxis]


def adaptive_resize(img, img_target):
    '''adaptivly resize an image according to a target image
    '''
    img = cv2.resize(img, (img_target.shape[1], img_target.shape[0]))
    return img


def rescale_fg(img, scale_factor=1.1):
    '''rescale the foreground

    Args:
        img (np.array<np.uint8>): the input image
    '''
    ori_h, ori_w = img.shape[0], img.shape[1]
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[0], img.shape[1]
    h_off = int((h - ori_h) / 2)
    w_off = int((w - ori_w) / 2)
    img = img[h_off:h_off+ori_h, w_off:w_off+ori_w]
    return img


def shift_fg(img, dx=0, dy=0):
    '''shift the foreground image according to position bias

    Args:
        img (np.array<np.uint8>): the input image
    '''
    height, width = img.shape[:2]
    trans_m = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, trans_m, (width, height))
    return img


def pad_resize(img, target_size):
    """reisze and pad the image to the target size.

    Args:
        img (np.array<np.uint8>): the input image
        target_size (Tuple[int]): shape (2,), the target (height, width)

    Returns:
        img (np.array<np.uint8>): the image after resizing and padding
        ratio (float): the resized ratio, new size / original size
    """
    target_h, target_w = target_size
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    if float(h) / w > float(target_h) / target_w:
        new_h = target_h
        ratio = float(target_h) / h
        new_w = int(float(target_h) * w / h)
        pad_w = target_w - new_w
        pad_h = 0
    else:
        new_w = target_w
        ratio = float(target_w) / w
        new_h = int(float(target_w) * h / w)
        pad_w = 0
        pad_h = target_h - new_h
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return img, ratio


def inv_pad_resize(img, ori_size):
    """restore the resized and padded image to the original size.

    It is the inversion of the function "pad_resize".

    Args:
        img (np.array<np.uint8>): the resized and padded image
        ori_size (Tuple[int]): shape (2,), the original (height, width)

    Returns:
        img (np.array<np.uint8>): the original image
    """
    ori_h, ori_w = ori_size
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    if float(ori_h) / ori_w > float(h) / w:
        resized_h = h
        resized_w = int(float(h) * ori_w / ori_h)
        pad_w = w - resized_w
        pad_h = 0
    else:
        resized_w = w
        resized_h = int(float(w) * ori_h / ori_w)
        pad_w = 0
        pad_h = h - resized_h
    if pad_w > 0:
        img = img[:, :-pad_w]
    if pad_h > 0:
        img = img[:-pad_h]
    img = cv2.resize(img, (ori_w, ori_h))
    return img


def imnormalize(img,
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]),
                to_rgb=True):
    """normalize the image.

    It would normalized the image by
        (image - mean) / std
    And if to_rgb is set as True, the image would be convert from BGR to RGB

    Args:
        img (np.array<np.uint8>): the input image
        mean (np.array<float>): shape (3,), the means of each channel
        std (np.array<float>): shape (3,), the stds of each channel
        to_rgb (bool): whether to convert to RGB space or not

    Returns:
        img (np.array<np.uint8>): the nomalized image
    """
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    img = img.astype(np.float32) / 255.
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def get_target_size(h, w, target_long_side, division=1):
    """calculate the target size with a given target long side.

    Give an image with size (h, w), we want to resize its long side to
    "target_long_side" and keep the target size can be divided by "division".
    This function would help you to calculate the target size meet the
    constrains above.

    Args:
        h (int): the original height
        w (int): the original width
        target_long_side (int): target long side
        division (int): keep the target size divisible by "division"

    Returns:
        target_h (int): the target height
        target_w (int): the target width
    """
    if h > w:
        target_h = target_long_side
        target_w = int(float(target_long_side) * w / h)
        if target_w % division != 0:
            target_w = (target_w // division + 1) * division
    else:
        target_w = target_long_side
        target_h = int(float(target_long_side) * h / w)
        if target_h % division != 0:
            target_h = (target_h // division + 1) * division
    return target_h, target_w


def remove_blackborder(img, location=None):
    """To remove the blackborder of an image.

    Args:
        img (np.array<np.uint8>): the input image
        location (Tuple[int], optional): the location
            (top, left, bottom, right) of the blackborder. If the location is
            given, the function would remove the blackborder with the given
            location and return the image only. Otherwise, the function would
            detect the location of the blackborder, remove it and return both
            the image and the location.

    Returns:
        img (np.array<np.uint8>): the image without blackborder
        location (Tuple[int]): shape (4,), the location, i.e.
            (top, left, bottom, right), of the detected blackborder
    """
    h, w = img.shape[:2]
    if location is None:
        left, r, t, b = 0, w, 0, h
        for i in range(h):
            if img[i, :].sum() == 0:
                t += 1
        for i in range(w):
            if img[:, i].sum() == 0:
                left += 1
        for i in range(h - 1, -1, -1):
            if img[i, :].sum() == 0:
                b -= 1
        for i in range(w - 1, -1, -1):
            if img[:, i].sum() == 0:
                r -= 1
        if left >= r or t >= b:
            return img, (0, 0, h, w)
        return img[t:b, left:r], (t, left, b, r)
    else:
        t, left, b, r = location
        return img[t:b, left:r]


def add_blackborder(img, ori_size, location):
    """add blackborder to an image.

    It is the inversion of the function "remove_blackborder".

    Args:
        img (np.array<np.uint8>): the image without blackborder
        ori_size (Tuple[int]): shape (2,), (height, width) of
            the original image
        location (Tuple[int]): shape (4,), the location, i.e.
            (top, left, bottom, right), of the blackborder

    Returns:
        img_with_blackborder(np.array<np.uint8>): the image with blackborder
    """
    t, left, b, r = location
    ori_h, ori_w = ori_size
    if t == 0 and left == 0 and b == ori_h and t == ori_w:
        return img
    if img.ndim == 3:
        c = img.shape[-1]
        img_with_blackborder = np.zeros((ori_h, ori_w, c), np.uint8)
    else:
        img_with_blackborder = np.zeros((ori_h, ori_w), np.uint8)
    img_with_blackborder[t:b, left:r] = img
    return img_with_blackborder


def color_correct(img, alpha, bg_color, target_long_side=960, mean_exp=0.95):
    """transform to Lab color space to calculate the distance to the background
    for color correction.

    Args:
        img (np.array<np.uint8>): the input image
        alpha (np.array<np.uint8>): the alpha channel
        bg_color (np.array<np.uint8>): shape (3,),
            the color to remove, usually is the background color
        target_long_side (int): target long side to resize to
        mean_exp (float): after calculate the distance, the value of distance
            map would usually too small, the mean value of the foreground
            should be shift, mean_thr is the experation of the mean value to
            shift to

    Returns:
        alpha (np.array<np.uint8>): the alpha channel after color correction
    """
    h, w = img.shape[:2]
    target_h, target_w = get_target_size(h, w, target_long_side)
    img = cv2.resize(img, (target_w, target_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    bg_color = bg_color[np.newaxis, np.newaxis, ...]
    bg_color = cv2.cvtColor(bg_color, cv2.COLOR_BGR2Lab)
    img_tensor = torch.from_numpy(img).float() / 255.
    bg_tensor = torch.from_numpy(bg_color).float() / 255.
    dist = torch.sqrt(((img_tensor - bg_tensor)[:, :, 1:]**2).sum(dim=-1))
    dist = (dist - dist.min()) / (dist.max() - dist.min())
    alpha_tensor = torch.from_numpy(cv2.resize(alpha,
                                               (target_w, target_h))).float()
    while dist[(alpha_tensor > 0) & (dist > 0)].mean() < mean_exp:
        dist = torch.sqrt(dist)
    dist[alpha_tensor == 0] = 0
    dist = torch.nn.functional.interpolate(
        dist.unsqueeze(0).unsqueeze(0), (h, w)).squeeze()
    alpha = (torch.from_numpy(alpha).float() * dist).numpy()
    alpha = alpha.astype(np.uint8)
    return alpha
