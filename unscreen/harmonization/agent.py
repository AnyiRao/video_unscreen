import cv2
import numpy as np

from ..utils import get_target_size


class HarmonizationAgent():
    """A harmonization class with some tool to harmonize foreground and
    background."""

    def __init__(self):
        pass

    def get_means(self, img, mask=None, target_long_side=240):
        """get means of a region in an image.

        Args:
            img (np.array<uint8>): a 3-channel image
            mask (np.array<bool>, optional): a boolen mask to indicate region
                of interest, if set as None, mean of the whole image would be
                calculated.

        Returns:
            means (np.array<float>): shape (3,), mean of each channel
        """
        h, w = img.shape[:2]
        target_h, target_w = get_target_size(h, w, target_long_side)
        img = cv2.resize(img, (target_w, target_h))
        if mask is None:
            samples = img.reshape(-1, 3)
        else:
            mask = cv2.resize(
                mask.astype(np.uint8) * 255, (target_w, target_h))
            mask = mask > 0
            samples = img[mask].reshape(-1, 3)
        means = samples.mean(axis=0).astype(np.float)
        return means

    def foreground_toning(self,
                          fg,
                          bg,
                          alpha,
                          toning_ratio=(0.5, 0.05, 0.05),
                          max_shift=15,
                          min_shift=-30):
        """toning foreground accroding to background.

        This function would shift the means of foreground to make it near to
        the means of background in the Lab color space.

        Args:
            fg (np.array<uint8>): the foreground image
            bg (np.array<uint8>): the background image
            alpha (np.array<uint8>): the alpha map
            toning_ratio (Tuple(float)): shape (3,)
                the ratio for each channel to shift
            max_shift (int): the upper bound of shifting
            min_shift (int): the lower bound of shifting

        Returns:
            fg_new (np.array<uint8>): the foreground image after toning
        """
        fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB)
        fg_means = self.get_means(fg_lab.copy(), alpha > 0)
        bg_means = self.get_means(bg_lab.copy())
        fg_lab = fg_lab.astype(np.float)
        for i in range(3):
            shift = toning_ratio[i] * (bg_means[i] - fg_means[i])
            shift = max(min(shift, max_shift), min_shift)
            fg_lab[:, :, i] = fg_lab[:, :, i] + shift
        fg_lab = np.clip(fg_lab, 0, 255)
        fg_lab = fg_lab.astype(np.uint8)
        fg_new = cv2.cvtColor(fg_lab, cv2.COLOR_LAB2BGR)
        # fg_new[alpha == 0] = fg[alpha == 0]
        return fg_new

    def alpha_smoothing(self, alpha, iters=3, ksize=3, target_long_side=1920):
        """smooth the alpha map.

        Args:
            alpha (np.array<uint8>): the alpha map
            iters (int): the iterations to run smoothing
            ksize (int): the kernel size of smoothing

        Returns:
            alpha_new (np.array<uint8>): the alpha map after smoothing
        """
        alpha_new = alpha.copy()
        h, w = alpha_new.shape[:2]
        target_h, target_w = get_target_size(h, w, target_long_side)
        alpha_new = cv2.resize(alpha_new, (target_w, target_h))
        for _ in range(iters):
            alpha_new = cv2.boxFilter(alpha_new, -1, (ksize, ksize))
        alpha_new = cv2.resize(alpha_new, (w, h))
        return alpha_new

    def background_blurring(self, bg, iters=3, ksize=3, target_long_side=480):
        """blur the background.

        Args:
            bg (np.array<uint8>): the background image
            iters (int): the iterations to run blurring
            ksize (int): the kernel size of blurring
        """
        bg_blur = bg.copy()
        h, w = bg_blur.shape[:2]
        target_h, target_w = get_target_size(h, w, target_long_side)
        bg_blur = cv2.resize(bg_blur, (target_w, target_h))
        for _ in range(iters):
            bg_blur = cv2.boxFilter(bg_blur, -1, (ksize, ksize))
        bg_blur = cv2.resize(bg_blur, (w, h))
        return bg_blur
