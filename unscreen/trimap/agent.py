import cv2
import numpy as np

from ..utils import dilate_mask, erode_mask, get_target_size, is_pixel_inrange


class TrimapAgent():
    """to generate trimap.

    Args:
        input_long_side (int): long side of the input image would be resize to
        kernelsize (int): the kernel size for dilation and erosion,
            the larger the kernelsize is, the larger unknown region would be
        iters (int): the iterations for dilation and erosion
            the larger the iters is, the larger the unknown region would be
        color_winsize (Tuple<int>), shape (3,),
            the window size of HSV channel when run "is_pixel_inrange", pixels
            in the range (bg_color-winsize//2, bg_color+winsize//2) would be
            taken as unknown in trimap

    Attributes:
        see Args
    """

    def __init__(self,
                 input_long_side=960,
                 kernelsize=3,
                 iters=5,
                 color_winsize=(10, 100, 180)):
        self.kernelsize = kernelsize
        self.iters = iters
        self.input_long_side = input_long_side
        self.color_winsize = color_winsize

    def generate_trimap(self, mask):
        """to generate trimap based on a mask.

        This function would dilate and erode the mask, and the inconsistent
        region between the dilated and eroded masks would be taken as unknown
        in the trimap

        Args:
            mask (np.array<uint8>): the input mask

        Returns:
            trimap (np.array<uint8>): the trimap,
                a one channel image with three possible values,
                0 for background, 128 for unknown, and 255 for foreground
        """
        ori_h, ori_w = mask.shape
        input_h, input_w = get_target_size(ori_h, ori_w, self.input_long_side)
        mask = cv2.resize(
            mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
        trimap = np.ones((input_h, input_w), np.uint8) * 128
        dilated_mask = dilate_mask(mask, self.kernelsize, self.iters)
        eroded_mask = erode_mask(mask, self.kernelsize, self.iters)
        trimap[eroded_mask > 127] = 255
        trimap[dilated_mask < 128] = 0
        trimap = cv2.resize(trimap, (ori_w, ori_h), cv2.INTER_NEAREST)
        trimap[np.logical_and(trimap > 0, trimap < 255)] = 128
        return trimap

    def generate_trimap_withbg(self, mask, img, bgimg):
        """to generate trimap based on both mask and a estimated background
        image.

        This function would fisrt estimate a background mask, named bgmask
        here, based on the img and the bgimg. To be specific, the pixels of the
        img would be taken as background if their values in a range of
        (bg-winsize//2, bg+winsize//2). Then if the estimated bgmask get
        consist with the input mask, they would be ensembled as a
        ensembled_mask. Finally the ensembled_mask would be used to generate
        the trimap using the "generate_trimap" fucntion.

        Args:
            mask (np.array<uint8>): the mask to indicate the foregound
            img (np.array<uint8>): the original image
            bgimg (np.array<uint8>): shape (h, w, 3) or (3,)
                the estimated background, can be an image with the same shape
                to the mask or a the values of each channel if it is a pure
                color background.

        Returns:
            trimap (np.array<uint8>): the trimap,
                a one channel image with three possible values,
                0 for background, 128 for unknown, and 255 for foreground
        """
        if (mask > 0).sum() == 0:
            return mask
        bgmask = is_pixel_inrange(img, bgimg, self.color_winsize)
        fuzzy_area = (mask > 0) & (bgmask > 0)
        # if the estimated bgmask can not be consensus to the input mask
        # just trust the input mask without ensemble
        if float(fuzzy_area.sum()) / (mask > 0).sum() > 0.1:
            trimap = self.generate_trimap(mask)
        else:
            ensembled_mask = mask.copy()
            ensembled_mask[fuzzy_area] = 0
            trimap = self.generate_trimap(ensembled_mask)
            trimap[fuzzy_area] = 128
        return trimap

    def forward(self, *args, **kwargs):
        """main function of the TrimapAgent, given a mask it would output a
        trimapmap, which would be used for matting.

        There are two methods to get a trimap, the first one is based on the
        mask only and the second one is based on the mask, the origianl image
        and the estimated background image.

        Args:
            mask (np.array<uint8>): the mask to indicate the foregound
            img (np.array<uint8>, optional): the original image
            bgimg (np.array<uint8>, optional): shape (h, w, 3) or (3,)
                the estimated background, can be an image with the same shape
                to the mask or a the values of each channel if it is a pure
                color background.

        Returns:
            trimap (np.array<uint8>): the trimap,
                a one channel image with three possible values,
                0 for background, 128 for unknown, and 255 for foreground
        """
        if len(args) > 2:
            trimap = self.generate_trimap_withbg(*args, **kwargs)
        else:
            trimap = self.generate_trimap(*args, **kwargs)
        return trimap
