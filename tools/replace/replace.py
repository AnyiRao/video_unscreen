import argparse
import os
import os.path as osp
import pdb
import sys
import time
from glob import glob

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

from unscreen.utils import (adaptive_resize, get_center, rescale_fg,
                            return_date, shift_fg)


def get_dx_dy(vid):
    # this is the same as the one computed in comp_dx_dy.
    if vid == "test3":
        return -1.725405405405406, 8.1281081081081081
    elif vid == "test5":
        return -0.2072, -2.6028
    else:
        return None, None


def comp_dx_dy(args, numframes):
    dst2src_dx_list, dst2src_dy_list = [], []
    for fid in tqdm(range(numframes)):
        dst_mask = mmcv.imread(osp.join(args.tgt_data_dir, "alphamask_{:06d}.jpg".format(fid)))
        dst_center = get_center(dst_mask)
        src_mask = mmcv.imread(osp.join(args.src_data_dir, "alphamask_{:06d}.jpg".format(fid)))
        src_mask = adaptive_resize(src_mask, dst_mask)
        src_center = get_center(src_mask)
        dst2src_dx = np.array(src_center)[0] - np.array(dst_center)[0]
        dst2src_dy = np.array(src_center)[1] - np.array(dst_center)[1]
        dst2src_dx_list.append(dst2src_dx)
        dst2src_dy_list.append(dst2src_dy)
    dx_mean, dy_mean = np.mean(dst2src_dx_list), np.mean(dst2src_dy_list)
    return dx_mean, dy_mean


def run(args):
    print("Processing", args)
    framepaths = sorted(glob(osp.join(args.tgt_data_dir, 'fg_*.jpg')))
    numframes = len(framepaths)
    assert numframes > 0

    src_vid = args.src
    os.makedirs(args.dst_data_dir, exist_ok=True)

    dx_mean, dy_mean = get_dx_dy(src_vid)
    if dx_mean is None or dy_mean is None:
        dx_mean, dy_mean = comp_dx_dy(args, numframes)
    print("Correspondence mean: ", dx_mean, dy_mean)

    src_bg_image = mmcv.imread(args.src_bg_image)
    for fid in tqdm(range(numframes)):
        dst_fg   = mmcv.imread(osp.join(args.tgt_data_dir, "fg_{:06d}.jpg".format(fid)))
        dst_mask = mmcv.imread(osp.join(args.tgt_data_dir, "alphamask_{:06d}.jpg".format(fid)))

        src_image = mmcv.imread(osp.join(args.src_data_dir, "frame_{:06d}.jpg".format(fid)))
        src_image = adaptive_resize(src_image, dst_fg)

        src_bg_image = adaptive_resize(src_bg_image.copy(), dst_fg)

        # merge fg and bg
        dst_fg_shift = shift_fg(dst_fg, dx=dx_mean, dy=dy_mean)
        dst_fg_shift = rescale_fg(dst_fg_shift, scale_factor=1.2)
        dst_mask_shift = shift_fg(dst_mask, dx=dx_mean, dy=dy_mean)
        dst_mask_shift = rescale_fg(dst_mask_shift, scale_factor=1.2)

        new_binary_mask = dst_mask_shift.astype(np.float) / 255
        res = dst_fg_shift.astype(np.float) * new_binary_mask + src_bg_image.astype(np.float) * (1 - new_binary_mask)
        res = res.astype(np.uint8)

        # save image
        res_fn = osp.join(args.dst_data_dir, "res_{:06d}.jpg".format(fid))
        compare_fn = osp.join(args.dst_data_dir, "compare_{:06d}.jpg".format(fid))
        mmcv.imwrite(res, res_fn)
        mmcv.imwrite(np.concatenate((src_image, res), axis=1), compare_fn)

    mmcv.frames2video(args.dst_data_dir,
                      osp.join(args.dst_vid_dir, "compare_{}_{}_{}.mp4".format(args.src, args.tgt, return_date())),
                      filename_tmpl="compare_{:06d}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default="test5")
    parser.add_argument('--tgt', type=str, default="out5")
    args = parser.parse_args()
    data_root = "../data/replace/edn"
    args.tgt = "out" + args.src[-1]
    args.src_img_dir = osp.join(data_root, "src_img", args.src+"_500")
    args.src_data_dir = osp.join(data_root, "unscreen_img", args.src)
    args.src_bg_image = osp.join(args.src_data_dir, "../bg/test5_case.jpg")
    args.tgt_data_dir = osp.join(data_root, "unscreenbg_img", args.tgt)
    args.dst_data_dir = osp.join(data_root, "merge_test_img", "{}_{}".format(args.src, args.tgt))
    args.dst_vid_dir = osp.join(data_root, "video")
    run(args)
