import numpy as np
import torch
import torch.nn as nn

from ..utils import get_target_size, imnormalize, inv_pad_resize, pad_resize
from .brs import (BasePredictor, Click, Clicker, DeepLabV3Plus, DistMapsModel,
                  FeatureBRSPredictor, ScaleBiasOptimizer, SepConvHead)


class ISegAgent():

    def __init__(self,
                 model_path,
                 with_brs=False,
                 input_long_side=800,
                 prob_thresh=0.5,
                 with_flip=True,
                 cuda_device=0):

        state_dict = torch.load(model_path, map_location='cpu')
        self.cuda_device = cuda_device
        self.input_long_side = input_long_side
        self.prob_thresh = prob_thresh
        self.with_flip = with_flip

        self.predictor = self.build_predictor(state_dict, with_brs)

    def build_predictor(self, state_dict, with_brs):
        # build network
        model = DistMapsModel(
            feature_extractor=DeepLabV3Plus(
                backbone='resnet50',
                ch=128,
                project_dropout=0,
                inference_mode=True),
            head=SepConvHead(
                1,
                in_channels=128,
                mid_channels=128 // 2,
                num_layers=2,
                norm_layer=nn.BatchNorm2d),
            use_rgb_conv=True,
            norm_layer=nn.BatchNorm2d)
        model.load_state_dict(state_dict, strict=False)
        if self.cuda_device >= 0:
            device = torch.device(f'cuda:{self.cuda_device}')
        else:
            device = torch.device('cpu')
        model.to(device)
        model.eval()

        if with_brs:
            # build optimizer
            brs_opt_func_params = {'min_iou_diff': 1e-3}
            lbfgs_params_ = {
                'm': 20,
                'factr': 0,
                'pgtol': 1e-8,
                'maxfun': 20,
                'maxiter': 20 * 2
            }
            opt_functor = ScaleBiasOptimizer(
                prob_thresh=self.prob_thresh,
                with_flip=self.with_flip,
                optimizer_params=lbfgs_params_,
                **brs_opt_func_params)

            # build predictor
            predictor_params_ = {
                'optimize_after_n_clicks': 1,
                'net_clicks_limit': 20
            }
            insertion_mode = 'after_aspp'
            predictor = FeatureBRSPredictor(
                model,
                device=device,
                opt_functor=opt_functor,
                with_flip=self.with_flip,
                insertion_mode=insertion_mode,
                **predictor_params_)
        else:
            # build predictor
            predictor_params_ = {'net_clicks_limit': 20}
            predictor = BasePredictor(
                model,
                device=device,
                with_flip=self.with_flip,
                **predictor_params_)

        return predictor

    def preprocess(self, img, click_history):
        h, w = img.shape[:2]
        input_size = get_target_size(h, w, self.input_long_side, 1)
        img, ratio = pad_resize(img, input_size)
        clicker = Clicker(np.zeros(img.shape[:2], dtype=np.bool))
        img_tensor = torch.from_numpy(imnormalize(img).transpose(
            2, 0, 1)).unsqueeze(0)
        for record in click_history:
            is_positive, y, x = record[0], int(record[1] * ratio), int(
                record[2] * ratio)
            click = Click(is_positive=is_positive, coords=(y, x))
            clicker._add_click(click)
        return img_tensor, clicker

    def forward(self, img, click_history):
        ori_size = img.shape[:2]
        img_tensor, clicker = self.preprocess(img, click_history)
        if self.cuda_device >= 0:
            img_tensor = img_tensor.cuda(self.cuda_device)
        self.predictor.set_input_image(img_tensor)
        pred = self.predictor.get_prediction(clicker)
        mask = self.postprocess(pred, ori_size)
        return mask

    def postprocess(self, pred, ori_size):
        pred = inv_pad_resize(pred, ori_size)
        mask = (pred > self.prob_thresh).astype(np.uint8) * 255
        return mask
