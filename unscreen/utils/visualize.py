"""some functions for visualization."""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def fuse_fgbg(fg, bg, mask):
    """to fuse the foreground from one image and the background from another
    image.

    Args:
        fg (np.array<uint8>): image to get foreground
        bg (np.array<uint8>): image to get background
        mask (np.array<uint8>): mask of the foreground,
            255 for foreground and 0 for background

    Returns:
        fused_img (np.array<uint8>): the fused image
    """
    fg_alpha = mask.astype(np.float)[..., np.newaxis] / 255
    fused_img = fg_alpha * fg.astype(np.float) + (1 - fg_alpha) * bg.astype(
        np.float)
    fused_img = fused_img.astype(np.uint8)
    return fused_img


def get_roi(img, mask):
    """get region of interest in an image.

    This function would set the region of interest as the orgianl value and the
    other regions as black, i.e. pixel with value (0, 0, 0)

    Args:
        img (np.array<uint8>): image to get foreground
        mask (np.array<uint8>): mask of the ROI,
            255 for ROI and 0 for others

    Returns:
        img_roi (np.array<uint8>): image with non-ROI set as black
    """
    img_roi = img.astype(np.float) * (
        mask.astype(np.float)[..., np.newaxis] / 255)
    img_roi = img_roi.astype(np.uint8)
    return img_roi


def highlight_roi(img, mask):
    """hightlight the region of interest (with a red mask)

    Args:
        img (np.array<uint8>): image to get foreground
        mask (np.array<uint8>): mask of the ROI,
            255 for ROI and 0 for others

    Returns:
        img_roi (np.array<uint8>): image with ROI covered with red mask
    """
    img_roi = img.copy()
    ratio = 0.5
    img_roi[:, :, -1] = ratio * img_roi[:, :, -1].astype(
        np.float) + (1 - ratio) * mask.astype(np.float)
    img_roi[:, :, -1][mask == 0] = img[:, :, -1][mask == 0]
    img_roi = img_roi.astype(np.uint8)
    return img_roi


def tocolor(img):
    """transform an image to BGR space if it is gray scale.

    Args:
        img (np.array<uint8>): the input image
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def show(img, downsacle=1):
    """show an image, the image would be down sampled if needed.

    Args:
        img (np.array<uint8>): image to show
        downsacle (int): the scale to down sample the image
    """
    assert isinstance(downsacle, int)
    if downsacle != 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsacle, h // downsacle))
    cv2.imshow('result', img)
    cv2.waitKey()


def show_dist_hist(samples, num_hist=20):
    """plot the histogram of samples with one-dimension feature, usually it is
    used to show the probability vector or matrix in range [0, 1]

    Args:
        samples (np.array): the samples
        num_hist: number of histogram bars
    """
    hist, edges = np.histogram(samples, num_hist, range=(0, 1))
    x_range = (edges[:-1] + edges[1:]) / 2
    plt.bar(x_range, hist, width=0.8 / num_hist)
    plt.xlim([0, 1])
    plt.show()
