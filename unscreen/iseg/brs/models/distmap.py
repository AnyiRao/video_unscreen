import torch
import torch.nn as nn


class DistMapsModel(nn.Module):

    def __init__(self,
                 feature_extractor,
                 head,
                 norm_layer=nn.BatchNorm2d,
                 use_rgb_conv=True):
        super(DistMapsModel, self).__init__()

        if use_rgb_conv:
            self.rgb_conv = nn.Sequential(
                nn.Conv2d(in_channels=5, out_channels=8, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2),
                norm_layer(8),
                nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1),
            )
        else:
            self.rgb_conv = None

        self.dist_maps = DistMaps(norm_radius=260, spatial_scale=1.0)
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, image, points):
        coord_features = self.dist_maps(image, points)

        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
        else:
            c1, c2 = torch.chunk(coord_features, 2, dim=1)
            c3 = torch.ones_like(c1)
            coord_features = torch.cat((c1, c2, c3), dim=1)
            x = 0.8 * image * coord_features + 0.2 * image

        backbone_features = self.feature_extractor(x)
        instance_out = self.head(backbone_features[0])
        instance_out = nn.functional.interpolate(
            instance_out,
            size=image.size()[2:],
            mode='bilinear',
            align_corners=True)

        # return {'instances': instance_out}
        return instance_out

    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights, map_location='cpu')
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)

    def get_trainable_params(self):
        backbone_params = nn.ParameterList()
        other_params = nn.ParameterList()

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        return backbone_params, other_params


class DistMaps(nn.Module):

    def __init__(self, norm_radius, spatial_scale=1.0):
        super(DistMaps, self).__init__()
        self.xs = None
        self.coords_shape = None
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self._count = 0

    def get_coord_features(self, points, rows, cols, num_points):
        invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0

        row_array = torch.arange(
            start=0,
            end=rows,
            step=1,
            dtype=torch.float32,
            device=points.device)
        col_array = torch.arange(
            start=0,
            end=cols,
            step=1,
            dtype=torch.float32,
            device=points.device)

        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
        coords = torch.stack(
            (coord_rows, coord_cols),
            dim=0).unsqueeze(0).repeat(self.coords_shape[0], 1, 1, 1)

        add_xy = (points * self.spatial_scale).view(self.coords_shape[0],
                                                    points.size(1), 1, 1)
        coords.add_(-add_xy).div_(self.norm_radius * self.spatial_scale)
        coords.mul_(coords)

        coords[:, 0] += coords[:, 1]
        coords = coords[:, :1]

        coords[invalid_points, :, :, :] = 1e6

        coords = coords.view(-1, num_points, 1, rows, cols)
        coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
        coords = coords.view(-1, 2, rows, cols)
        coords.sqrt_().mul_(2).tanh_()
        return coords

    def forward(self, x, coords):
        num_points = coords.shape[1] // 2
        coords = coords.view(-1, 2)

        self.xs = x.shape
        self.coords_shape = coords.shape

        _, rows, cols = self.xs[0], self.xs[2], self.xs[3]
        return self.get_coord_features(coords, rows, cols, num_points)
