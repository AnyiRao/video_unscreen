import numpy as np
import torch
import torch.nn.functional as F

from ..utils import get_target_size, imnormalize, inv_pad_resize, pad_resize
from .model import STM
import pdb


class STMAgent():
    """Space-Time Memory Networks for video object segmentation.

    Reference Paper: Video Object Segmentation using Space-Time Memory Networks
        http://openaccess.thecvf.com/content_ICCV_2019/html/
        Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks
        _ICCV_2019_paper.html
    The code is modified from this repo: https://github.com/seoungwugoh/STM

    Args:
        model_path (str): the path of the weights of the deep model
        memory_step (int): the step of frames to update memory
            for example, if memory_step=5, it means that we would update the
            memory each 5 frames.
        memory_capacity (int): the capacity of the memory bank
        input_long_side (int): the long side of the input frame would be resize
            to
        cuda_device (int): the device to run the model,
            if set as negative, it would run in CPU

    Attributes:
        model (nn.Module): the deep model
        memory_step (int): the step of frames to update memory
            for example, if memory_step=5, it means that we would update the
            memory each 5 frames.
        memory_capacity (int): the capacity of the memory bank
        input_long_side (int): the long side of the input frame would be resize
            to
        cuda_device (int): the device to run the model,
            if set as negative, it would run in CPU
        division (int): the division that need to meet when resize and pad the
            frame here we set as a fixed number 16
    """

    def __init__(self,
                 model_path,
                 memory_step=5,
                 memory_capacity=10,
                 input_long_side=960,
                 cuda_device=0):
        self.model = STM()
        self.cuda_device = cuda_device
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        if self.cuda_device >= 0:
            self.model.cuda(self.cuda_device)
        self.model.eval()

        self.division = 16
        self.input_long_side = input_long_side
        self.memory_step = memory_step
        self.memory_capacity = memory_capacity

    def preprocess(self, framelist, mask0):
        """preprocessing, including resize, pad, normalize and to_tensor.

        Args:
            framelist (List[np.array<uint8>]): a list of N frames, in BGR color
                space
            mask0 (np.array<uint8>): the mask of first frame

        Returns:
            frames_tensor (toch.Tensor<float>): shape
                (1, 3, N_frames, input_h, input_w) tensor of the input frames
            mask0_tensor (toch.Tensor<float>): shape
                (1, 2, 1, input_h, input_w) tensor of the mask of first frame,
                with 2 channels, the first one to indicate the background and
                the second one to indicate the foreground
            ori_size (Tuple[int]): the (height, width) the original frame
                before pad&resize
        """
        num_frames = len(framelist)
        h, w = mask0.shape
        input_size = get_target_size(h, w, self.input_long_side, self.division)
        input_h, input_w = input_size
        # process mask
        mask0, _ = pad_resize(mask0, input_size)
        mask0 = mask0 > 127
        mask0_array = np.empty((2, 1, input_h, input_w), dtype=np.uint8)
        mask0_array[0, 0] = (1 - mask0).astype(np.float32)  # background
        mask0_array[1, 0] = mask0.astype(np.float32)  # foreground
        mask0_tensor = torch.unsqueeze(torch.from_numpy(mask0_array), 0)
        # process frames
        frames_array = np.empty((3, num_frames, input_h, input_w),
                                dtype=np.float32)
        for i, frame in enumerate(framelist):
            frame, _ = pad_resize(frame, input_size)
            frames_array[:, i] = imnormalize(frame).transpose((2, 0, 1))
        frames_tensor = torch.unsqueeze(torch.from_numpy(frames_array), 0)
        return frames_tensor, mask0_tensor, (h, w)

    def inference(self, frames_tensor, mask0_tensor):
        """run video object segmentation to get the masks of the frames.

        The STM model would run memorize and segment.
        In meorization, the model would get the memory key and value given a
        frame and a mask.
        In segmentation, the model would get the mask given a frame and the
        memory keys and values of previous frames.
        Here are the shapes of some important tensors:
        preds_tensor: (1, 2, N_frames, input_h, input_w)
        keys: (1, 128, N_memory, input_h/16, input_w/16)
        values: (1, 512, N_memory, input_h/16, input_w/16)

        Args:
            frames_tensor (torch.Tensor<Float>): shape
                (1, 3, N_frames, input_h, input_w),
                where N_frames is the number of frames
            mask0_tensor (torch.Tensor<Float>): shape
                (1, 2, 1, input_h, input_w), mask of the first frame

        Returns:
            preds_array (np.array<np.float>): shape
                (2, N_frames, input_h, input_w) the predicted results, which
                has two channels (first dimension), the first one is the score
                of background and the second one is the score of foreground
        """
        num_frames = frames_tensor.size(2)
        b, c, _, h, w = mask0_tensor.size()
        preds_tensor = torch.zeros(b, c, num_frames, h, w)
        preds_tensor[:, :, 0] = mask0_tensor[:, :, 0]

        for t in range(1, num_frames):
            # memorize
            with torch.no_grad():
                if self.cuda_device >= 0:
                    prev_key, prev_value = self.model(
                        frames_tensor[:, :, t - 1].cuda(self.cuda_device),
                        preds_tensor[:, :, t - 1].cuda(self.cuda_device))
                    prev_key = prev_key.detach().unsqueeze(dim=2).cpu()
                    prev_value = prev_value.detach().unsqueeze(dim=2).cpu()
                else:
                    prev_key, prev_value = self.model(
                        frames_tensor[:, :, t - 1], preds_tensor[:, :, t - 1])
                    prev_key = prev_key.detach().unsqueeze(dim=2)
                    prev_value = prev_value.detach().unsqueeze(dim=2)
            if t - 1 == 0:
                # only prev memory
                input_keys, input_values = prev_key, prev_value
            else:
                input_keys = torch.cat([keys, prev_key], dim=2)  # noqa
                input_values = torch.cat([values, prev_value], dim=2)  # noqa
                if input_keys.size(2) > self.memory_capacity:
                    input_keys = input_keys[:, :, 1:]
                    input_values = input_values[:, :, 1:]
            # segment
            with torch.no_grad():
                if self.cuda_device >= 0:
                    logit = self.model(
                        frames_tensor[:, :, t].cuda(self.cuda_device),
                        input_keys.cuda(self.cuda_device),
                        input_values.cuda(self.cuda_device))
                    logit = logit.detach().cpu()
                else:
                    logit = self.model(frames_tensor[:, :, t], input_keys,
                                       input_values).detach()
            preds_tensor[:, :, t] = F.softmax(logit, dim=1)
            # update
            if t - 1 % self.memory_step == 0:
                keys, values = input_keys, input_values  # noqa
        preds_array = preds_tensor[0].numpy()
        return preds_array

    def postprocess(self, preds_array, ori_size):
        """postprocessing.

        Postprocessing would do the following things:
        1. transpose the preds_array
        2. inverse pad and resize
        3. transform the output from a 2-channel prediction to a 1-channel mask

        Args:
            preds_array (np.array<np.float>): shape
                (2, N_frames, input_h, input_w) the predicted results, which
                has two channels (first dimension), the first one is the score
                of background and the second one is the score of foreground
            ori_size (Tuple[int]): the original size, i.e. (height, width), of
                the input frame

        Returns:
            masklist (List[np.array<uint8>]): list of the predicted mask for
                each frame
        """
        preds_array = preds_array.transpose((1, 2, 3, 0))
        numframes = preds_array.shape[0]
        masklist = []
        for i in range(numframes):
            pred_score = inv_pad_resize(preds_array[i], ori_size)
            pdb.set_trace()
            masklist.append(
                (np.argmax(pred_score, axis=-1) * 255).astype(np.uint8))
        return masklist

    def forward(self, framelist, mask0):
        """main function of STMAgent, given a list of frames and the mask of
        the first frame, it would predict masks of all frames.

        Args:
            framelist (List[np.array<uint8>]): a list of frames, in BGR color
                space
            mask0 (np.array<uint8>): the mask of first frame

        Returns:
            masklist (List[np.array<uint8>]): list of the predicted mask for
                each frame
        """
        frames_tensor, mask0_tensor, ori_size = self.preprocess(
            framelist, mask0)
        preds_array = self.inference(frames_tensor, mask0_tensor)
        masklist = self.postprocess(preds_array, ori_size)
        print('mask list')
        pdb.set_trace()
        return masklist
