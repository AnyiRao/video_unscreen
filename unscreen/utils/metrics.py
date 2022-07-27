"""some functions about evaluation metrics."""
import numpy as np
import pdb
import cv2
from .maskprocess import dilate_mask, erode_mask


def get_ious(alpha, pred_alpha):
    """get IoUs (Intersection over Union) for each class.

    Here we have only two classes, i.e. foreground and background

    Args:
        pred_alpha (np.array<uint8>): the pred_alphaicted mask
        alpha (np.array<uint8>): the ground-truth mask

    Returns:
        ious (np.array<float>): shape (2,)
            the ious of foreground and background
    """
    w, h = pred_alpha.shape[:2]
    ious = np.zeros(2, )
    fg_intersection = ((pred_alpha > 127) & (alpha > 127)).sum()
    fg_union = ((pred_alpha > 127) | (alpha > 127)).sum()
    if fg_union < w * h * 0.001:
        ious[0] = 1
    else:
        ious[0] = float(fg_intersection) / fg_union
    bg_intersection = ((pred_alpha < 128) & (alpha < 128)).sum()
    bg_union = ((pred_alpha < 128) | (alpha < 128)).sum()
    if bg_union < w * h * 0.001:
        ious[1] = 1
    else:
        ious[1] = float(bg_intersection) / bg_union
    return ious


def get_miou(alpha, pred_alpha):
    """get mean IoU of the pred_alphaiction.

    Args:
        pred_alpha (np.array<uint8>): the pred_alphaicted mask
        alpha (np.array<uint8>): the ground-truth mask

    Returns:
        miou (float): the mean IoU of foreground and background
    """
    ious = get_ious(alpha, pred_alpha)
    miou = ious.mean()
    return miou


def get_sad(alpha, pred_alpha):
    """get SAD (Sum of Absolute Distance) of the pred_alphaiction.

    Here the SAD is normalized by the image size

    Args:
        pred_alpha (np.array<uint8>): the pred_alphaicted mask
        alpha (np.array<uint8>): the ground-truth mask

    Returns:
        sad (float): the sad of the pred_alphaiction
    """
    sad = np.abs(
        (pred_alpha.astype(np.float) - alpha.astype(np.float)) / 255.0).sum()
    sad /= np.sqrt(pred_alpha.shape[0] * pred_alpha.shape[1])
    return sad


def get_roi_sad(alpha, pred_alpha):
    """get SAD (Sum of Absolute Distance) of the pred_alphaiction in the region of
    interest, i.e. the boundary of the groundtruth mask.

    This metric focus more on the details of the pred_alphaiction, which is also
    the hard region for matting.
    Here the SAD is normalized by the size of the ROI.

    Args:
        pred_alpha (np.array<uint8>): the pred_alphaicted mask
        alpha (np.array<uint8>): the ground-truth mask

    Returns:
        sad (float): the sad of the pred_alphaiciton in the boundary
    """
    dilated_mask = dilate_mask(alpha)
    eroded_mask = erode_mask(alpha)
    roi = (dilated_mask > 0) ^ (eroded_mask > 0)
    sad = np.abs((pred_alpha[roi].astype(np.float) - alpha[roi].astype(np.float)) /
                 255.0).sum()
    sad = sad / float(roi.sum())
    return sad


def get_mse(alpha, pred_alpha):
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    mse = ((pred_alpha - alpha)**2).sum()
    return mse / 1000


def get_gradient_error(alpha, pred_alpha, sigma=1.4):
    """Gradient error for evaluating alpha matte pred_alphaiction.

    Args:
        alpha (ndarray): Ground-truth alpha matte.
        pred_alpha (ndarray): pred_alphaicted alpha matte.
        sigma (float): Standard deviation of the gaussian kernel. Default: 1.4.
    """
    alpha = alpha.astype(np.float64)
    pred_alpha = pred_alpha.astype(np.float64)
    alpha_normed = np.zeros_like(alpha)
    pred_alpha_normed = np.zeros_like(pred_alpha)
    cv2.normalize(alpha, alpha_normed, 1., 0., cv2.NORM_MINMAX)
    cv2.normalize(pred_alpha, pred_alpha_normed, 1., 0., cv2.NORM_MINMAX)

    alpha_grad = gauss_gradient(alpha_normed, sigma).astype(np.float32)
    pred_alpha_grad = gauss_gradient(pred_alpha_normed,
                                     sigma).astype(np.float32)

    grad_loss = ((alpha_grad - pred_alpha_grad)**2).sum()
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return grad_loss / 1000


def get_connectivity(alpha, pred_alpha, step=0.1):
    """Connectivity error for evaluating alpha matte prediction.

    Args:
        alpha (ndarray): Ground-truth alpha matte with shape (height, width).
            Value range of alpha is [0, 255].
        pred_alpha (ndarray): pred_alphaicted alpha matte with shape (height, width).
            Value range of pred_alpha is [0, 255].
        step (float): Step of threshold when computing intersection between
            `alpha` and `pred_alpha`.
    """
    alpha = alpha.astype(np.float32) / 255
    pred_alpha = pred_alpha.astype(np.float32) / 255

    height, width = alpha.shape[:2]
    thresh_steps = np.arange(0, 1 + step, step)
    round_down_map = -np.ones_like(alpha)
    for i in range(1, len(thresh_steps)):
        alpha_thresh = alpha >= thresh_steps[i]
        pred_alpha_thresh = pred_alpha >= thresh_steps[i]
        intersection = (alpha_thresh & pred_alpha_thresh).astype(np.uint8)

        # connected components
        _, output, stats, _ = cv2.connectedComponentsWithStats(
            intersection, connectivity=4)
        # start from 1 in dim 0 to exclude background
        size = stats[1:, -1]

        # largest connected component of the intersection
        omega = np.zeros_like(alpha)
        if len(size) != 0:
            max_id = np.argmax(size)
            # plus one to include background
            omega[output == max_id + 1] = 1

        mask = (round_down_map == -1) & (omega == 0)
        round_down_map[mask] = thresh_steps[i - 1]
    round_down_map[round_down_map == -1] = 1
    alpha_diff = alpha - round_down_map
    pred_alpha_diff = pred_alpha - round_down_map
    # only calculate difference larger than or equal to 0.15
    alpha_phi = 1 - alpha_diff * (alpha_diff >= 0.15)
    pred_alpha_phi = 1 - pred_alpha_diff * (pred_alpha_diff >= 0.15)

    connectivity_error = np.sum(
        np.abs(alpha_phi - pred_alpha_phi))
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return connectivity_error / 1000


def gaussian(x, sigma):
    """Gaussian function.

    Args:
        x (array_like): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Return:
        ndarray or scalar: Gaussian value of `x`.
    """
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def dgaussian(x, sigma):
    """Gradient of gaussian.

    Args:
        x (array_like): The independent variable.
        sigma (float): Standard deviation of the gaussian function.

    Return:
        ndarray or scalar: Gradient of gaussian of `x`.
    """
    return -x * gaussian(x, sigma) / sigma**2


def gauss_filter(sigma, epsilon=1e-2):
    """Gradient of gaussian.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
        epsilon (float): Small value used when calculating kernel size.
            Default: 1e-2.

    Return:
        tuple[ndarray]: Gaussian filter along x and y axis.
    """
    half_size = np.ceil(
        sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
    size = np.int(2 * half_size + 1)

    # create filter in x axis
    filter_x = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            filter_x[i, j] = gaussian(i - half_size, sigma) * dgaussian(
                j - half_size, sigma)

    # normalize filter
    norm = np.sqrt((filter_x**2).sum())
    filter_x = filter_x / norm
    filter_y = np.transpose(filter_x)

    return filter_x, filter_y


def gauss_gradient(img, sigma):
    """Gaussian gradient.

    From https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/8060/versions/2/previews/gaussgradient/gaussgradient.m/index.html  # noqa

    Args:
        img (ndarray): Input image.
        sigma (float): Standard deviation of the gaussian kernel.

    Return:
        ndarray: Gaussian gradient of input `img`.
    """
    filter_x, filter_y = gauss_filter(sigma)
    img_filtered_x = cv2.filter2D(
        img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
    img_filtered_y = cv2.filter2D(
        img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
    return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
