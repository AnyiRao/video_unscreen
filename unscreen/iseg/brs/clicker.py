from collections import namedtuple

import numpy as np

Click = namedtuple('Click', ['is_positive', 'coords'])


class Clicker(object):

    def __init__(self,
                 gt_mask,
                 init_clicks=None,
                 click_radius=1,
                 ignore_label=-1):
        self.gt_mask = gt_mask == 1
        self.not_ignore_mask = gt_mask != ignore_label
        self.height, self.width = gt_mask.shape
        self.radius = click_radius
        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self._add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def get_clicks_maps(self):
        pos_clicks_map = self.pos_clicks_map.copy()
        neg_clicks_map = self.neg_clicks_map.copy()

        if self.radius > 0:
            pos_clicks_map = np.zeros_like(pos_clicks_map, dtype=np.bool)
            neg_clicks_map = pos_clicks_map.copy()

            for click in self.clicks_list:
                y, x = click.coords
                y1, x1 = y - self.radius, x - self.radius
                y2, x2 = y + self.radius + 1, x + self.radius + 1

                if click.is_positive:
                    pos_clicks_map[y1:y2, x1:x2] = True
                else:
                    neg_clicks_map[y1:y2, x1:x2] = True

        pos_clicks_map = pos_clicks_map.astype(np.float32)
        neg_clicks_map = neg_clicks_map.astype(np.float32)

        return pos_clicks_map, neg_clicks_map

    def _add_click(self, click):
        coords = click.coords
        if click.is_positive:
            self.num_pos_clicks += 1
            self.pos_clicks_map[coords[0], coords[1]] = True

        else:
            self.num_neg_clicks += 1
            self.neg_clicks_map[coords[0], coords[1]] = True

        self.not_clicked_map[coords[0], coords[1]] = False
        self.clicks_list.append(click)

    def reset_clicks(self):
        self.pos_clicks_map = np.zeros_like(self.gt_mask, dtype=np.bool)
        self.neg_clicks_map = np.zeros_like(self.gt_mask, dtype=np.bool)
        self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def __len__(self):
        return len(self.clicks_list)
