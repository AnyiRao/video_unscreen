import numpy as np
import torch
import torch.nn.functional as F

from ..utils import get_target_size, imnormalize, inv_pad_resize, pad_resize
from .model import UNet


class VMattingAgent():

    def __init__(self, model_path, input_long_side=960, cuda_device=0):
        """Video-based Matting, to predict the alpha values of the unknown
        region given a frame, a trimap and the alpha map of last frame.

        Args:
            model_path (str): the path of the weights of the deep model
            input_long_side (int): the long side of the input frame would be
                resize to
            cuda_device (int): the device to run the model,
                if set as negative, it would run in CPU

        Attributes:
            model (nn.Module): the deep model, only support unet now
            cuda_device (int): the device to run the model
            input_long_side (int): the long side of the input frame would be
                resize to
            division (int): the division that need to meet when resize and pad
                the frame here we set as a fixed number 32
        """
        self.model = UNet()
        weights = torch.load(model_path, map_location='cpu')['state_dict']
        self.model.load_state_dict(weights)
        self.cuda_device = cuda_device
        if self.cuda_device >= 0:
            self.model.cuda(self.cuda_device)
        self.model.eval()

        self.division = 32
        if input_long_side % self.division != 0:
            input_long_side = (input_long_side // self.division +
                               1) * self.division
        self.input_long_side = input_long_side

    def to_tensor(self, img, alpha_pre, trimap):
        """transform the frame and trimap to torch.tensor.

        Args:
            img (np.array<uint8>): the input frame
            alpha_pre (np.array<uint8>): alpha map of last frame
            trimap (np.array<uint8>): the trimap, 1-channel image

        Returns:
            img (torch.Tensor<float>): shape (1, 3, input_h, input_w)
            alpha_pre_tensor (torch.Tensor<float>): shape
                (1, 1, input_h, input_w)
            trimap (torch.Tensor<float>): shape (1, 3, input_h, input_w),
                each channel is a one-hot map to indicate whether it belongs to
                background, unknown or foreground.
        """
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).float()
        img = torch.unsqueeze(img, 0)

        alpha_pre = torch.from_numpy(alpha_pre).float()
        alpha_pre = torch.unsqueeze(alpha_pre, 0)
        alpha_pre = torch.unsqueeze(alpha_pre, 0)

        trimap[np.logical_and(trimap > 0, trimap < 255)] = 1
        trimap[trimap == 255] = 2
        trimap = F.one_hot(torch.from_numpy(trimap).long(), num_classes=3)
        trimap = trimap.permute(2, 0, 1).float()
        trimap = torch.unsqueeze(trimap, 0)

        return img, alpha_pre, trimap

    def preprocess(self, img, alpha_pre, trimap):
        """preprocessing, including resize, pad, normalize and to_tensor.

        Args:
            img (np.array<uint8>): the input frame, in BGR color space
            alpha_pre (np.array<uint8>): alpha map of last frame
            trimap (np.array<uint8>): the trimap, a one channel image with
                three possible values, where 0 for background, 128 for unknown,
                and 255 for foreground
        Retures:
            img_tensor (torch.Tesnor<float>): shape (1, 3, input_h, input_w)
            alpha_pre_tensor (torch.Tensor<float>):
                shape (1, 1, input_h, input_w)
            trimap (torch.Tensor<float>): shape (1, 3, input_h, input_w),
                each channel is a one-hot map to indicate whether it belongs to
                background, unknown or foreground
            ori_trimap (np.array<uint8>): shape (h, w),
                save the original trimap for other operations
        """
        ori_trimap = trimap.copy()
        h, w = trimap.shape
        input_size = get_target_size(h, w, self.input_long_side, self.division)
        img, _ = pad_resize(img, input_size)
        trimap, _ = pad_resize(trimap, input_size)
        alpha_pre, _ = pad_resize(alpha_pre, input_size)
        alpha_pre = alpha_pre.astype(np.float) / 255.

        img = imnormalize(img)
        img_tensor, alpha_pre_tensor, trimap_tensor = self.to_tensor(
            img, alpha_pre, trimap)
        return img_tensor, alpha_pre_tensor, trimap_tensor, ori_trimap

    def postprocess(self, pred_alpha, trimap):
        """postprocessing to get the final predicted alpha.

        Postprocessing do the following two things:
        1. inverse pad and resise
        2. set the foreground and background the same as the original trimap,
        Note that although matting predict the whole alpha map, we only
        use the prediction of the unknown region, the foreground and background
        would be set the same to the orginal trimap.

        Args:
            pred_alpha (np.array<float>): shape (input_h, input_w),
                the output of the model, the scale is [0, 1]
            trimap (np.array<uint8>): shape (h, w),
                the input trimap

        Returns:
            pred_alpha (np.array<uint8>): shape (h, w),
                the final predicted alpha map
        """
        pred_alpha = inv_pad_resize(pred_alpha, trimap.shape)
        pred_alpha[trimap == 0] = 0.
        pred_alpha[trimap == 255] = 1.
        pred_alpha = (pred_alpha * 255).astype(np.uint8)
        return pred_alpha

    def forward(self, img, alpha_pre, trimap):
        """main function of the VMattingAgent, given a frame, a trimap, and the
        alpha map of last frame, it would predict the alpha map of this frame.

        Args:
            img (np.array<uint8>): shape (h, w, 3),
                the input frame, in BGR color space
            alpha_pre (np.array<uint8>): shape (h, w)
                alpha map of last frame
            trimap (np.array<uint8>): shape (h, w),
                the input trimap, a one channel image with three possible
                values, where 0 for background, 128 for unknown, and 255 for
                foreground

        Returns:
            pred_alpha (np.array<uint8>): shape (h, w),
                the final predicted alpha map
        """
        img_tensor, alpha_pre_tensor, trimap_tensor, trimap = self.preprocess(
            img, alpha_pre, trimap)
        if self.cuda_device >= 0:
            img_tensor = img_tensor.cuda(self.cuda_device)
            trimap_tensor = trimap_tensor.cuda(self.cuda_device)
            alpha_pre_tensor = alpha_pre_tensor.cuda(self.cuda_device)
        with torch.no_grad():
            pred_alpha = self.model(img_tensor, alpha_pre_tensor,
                                    trimap_tensor)
        pred_alpha = pred_alpha.cpu().numpy().squeeze()
        pred_alpha = self.postprocess(pred_alpha, trimap)
        return pred_alpha
