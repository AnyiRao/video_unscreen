import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b

from .transforms import AddHorizontalFlip, SigmoidForPred


class BasePredictor(object):

    def __init__(self,
                 net,
                 device,
                 net_clicks_limit=None,
                 with_flip=False,
                 **kwargs):
        self.net = net
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device

        self.transforms = []
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image_nd):
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)

    def get_prediction(self, clicker):
        clicks_list = clicker.get_clicks()
        clicks_maps = self._get_clicks_maps(clicker)

        # !!!
        (image_nd, clicks_lists, clicks_maps,
         is_image_changed) = self.apply_transforms(self.original_image,
                                                   [clicks_list], clicks_maps)

        pred_logits = self._get_prediction(image_nd, clicks_lists, clicks_maps,
                                           is_image_changed)
        prediction = F.interpolate(
            pred_logits,
            mode='bilinear',
            align_corners=True,
            size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        return prediction.detach().cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, clicks_maps,
                        is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)

    def apply_transforms(self, image_nd, clicks_lists, clicks_maps=None):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists, clicks_maps = t.transform(
                image_nd, clicks_lists, clicks_maps)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, clicks_maps, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [
            sum(x.is_positive for x in clicks_list)
            for clicks_list in clicks_lists
        ]
        num_neg_clicks = [
            len(clicks_list) - num_pos
            for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
        ]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [
                click.coords for click in clicks_list if click.is_positive
            ]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [
                (-1, -1)
            ]

            neg_clicks = [
                click.coords for click in clicks_list if not click.is_positive
            ]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [
                (-1, -1)
            ]
            total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device).float()

    def _get_clicks_maps(self, clicker):
        raise NotImplementedError()


class BRSBasePredictor(BasePredictor):

    def __init__(self,
                 model,
                 device,
                 opt_functor,
                 optimize_after_n_clicks=1,
                 **kwargs):
        super().__init__(model, device, **kwargs)
        self.optimize_after_n_clicks = optimize_after_n_clicks
        self.opt_functor = opt_functor

        self.opt_data = None
        self.input_data = None

    def set_input_image(self, image_nd):
        super().set_input_image(image_nd)
        self.opt_data = None
        self.input_data = None

    def _get_clicks_maps(self, clicker):
        pos_map, neg_map = clicker.get_clicks_maps()
        return pos_map[np.newaxis, :], neg_map[np.newaxis, :]

    def _get_clicks_maps_nd(self, clicks_maps):
        pos_clicks_map, neg_clicks_map = clicks_maps
        with torch.no_grad():
            pos_clicks_map = torch.from_numpy(pos_clicks_map).to(self.device)
            neg_clicks_map = torch.from_numpy(neg_clicks_map).to(self.device)
            pos_clicks_map = pos_clicks_map.unsqueeze(1)
            neg_clicks_map = neg_clicks_map.unsqueeze(1)

        return pos_clicks_map, neg_clicks_map


class FeatureBRSPredictor(BRSBasePredictor):

    def __init__(self,
                 model,
                 device,
                 opt_functor,
                 insertion_mode='after_deeplab',
                 **kwargs):
        super().__init__(model, device, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
        self._c1_features = None

        if self.insertion_mode == 'after_deeplab':
            self.num_channels = model.feature_extractor.ch
        elif self.insertion_mode == 'after_c4':
            self.num_channels = model.feature_extractor.aspp_in_channels
        elif self.insertion_mode == 'after_aspp':
            self.num_channels = model.feature_extractor.ch + 32
        else:
            raise NotImplementedError

    def _get_prediction(self, image_nd, clicks_lists, clicks_maps,
                        is_image_changed):

        points_nd = self.get_points_nd(clicks_lists)
        pos_mask, neg_mask = self._get_clicks_maps_nd(clicks_maps)
        num_clicks = len(clicks_lists[0])
        bs = image_nd.shape[0] // 2 if self.with_flip else image_nd.shape[0]

        if self.opt_data is None or self.opt_data.shape[0] // (
                2 * self.num_channels) != bs:
            self.opt_data = np.zeros((bs * 2 * self.num_channels),
                                     dtype=np.float32)

        if (num_clicks <= self.net_clicks_limit or is_image_changed
                or self.input_data is None):
            self.input_data = self._get_head_input(image_nd, points_nd)

        def get_prediction_logits(scale, bias):
            scale = scale.view(bs, -1, 1, 1)
            bias = bias.view(bs, -1, 1, 1)
            if self.with_flip:
                scale = scale.repeat(2, 1, 1, 1)
                bias = bias.repeat(2, 1, 1, 1)

            scaled_backbone_features = self.input_data * scale
            scaled_backbone_features = scaled_backbone_features + bias

            if self.insertion_mode == 'after_c4':
                x = self.net.feature_extractor.aspp(scaled_backbone_features)
                x = F.interpolate(
                    x,
                    mode='bilinear',
                    size=self._c1_features.size()[2:],
                    align_corners=True)
                x = torch.cat((x, self._c1_features), dim=1)
                scaled_backbone_features = self.net.feature_extractor.head(x)
            elif self.insertion_mode == 'after_aspp':
                scaled_backbone_features = self.net.feature_extractor.head(
                    scaled_backbone_features)

            pred_logits = self.net.head(scaled_backbone_features)
            pred_logits = F.interpolate(
                pred_logits,
                size=image_nd.size()[2:],
                mode='bilinear',
                align_corners=True)
            return pred_logits

        self.opt_functor.init_click(get_prediction_logits, pos_mask, neg_mask,
                                    self.device)
        if num_clicks > self.optimize_after_n_clicks:
            opt_result = fmin_l_bfgs_b(
                func=self.opt_functor,
                x0=self.opt_data,
                **self.opt_functor.optimizer_params)
            self.opt_data = opt_result[0]

        with torch.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred_logits = self.opt_functor.best_prediction
            else:
                opt_data_nd = torch.from_numpy(self.opt_data).to(self.device)
                opt_vars, _ = self.opt_functor.unpack_opt_params(opt_data_nd)
                opt_pred_logits = get_prediction_logits(*opt_vars)

        return opt_pred_logits

    def _get_head_input(self, image_nd, points):
        with torch.no_grad():
            coord_features = self.net.dist_maps(image_nd, points)
            x = self.net.rgb_conv(torch.cat((image_nd, coord_features), dim=1))
            if (self.insertion_mode == 'after_c4'
                    or self.insertion_mode == 'after_aspp'):
                c1, _, _, c4 = self.net.feature_extractor.backbone(x)
                c1 = self.net.feature_extractor.skip_project(c1)

                if self.insertion_mode == 'after_aspp':
                    x = self.net.feature_extractor.aspp(c4)
                    x = F.interpolate(
                        x,
                        size=c1.size()[2:],
                        mode='bilinear',
                        align_corners=True)
                    x = torch.cat((x, c1), dim=1)
                    backbone_features = x
                else:
                    backbone_features = c4
                    self._c1_features = c1
            else:
                backbone_features = self.net.feature_extractor(x)[0]

        return backbone_features
