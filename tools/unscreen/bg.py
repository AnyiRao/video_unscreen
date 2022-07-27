import argparse
import json
import os
import os.path as osp
import pdb
import time
from glob import glob

import cv2
import numpy as np
import tqdm

from unscreen.binseg import HumanSegAgent as SegAgent
from unscreen.stm import STMAgent
from unscreen.trimap import TrimapAgent
from unscreen.utils import (dilate_mask, exist_foreground, get_bg, get_fg,
                            parallel_read_img, regionfill,
                            remove_invalid_objects, save_img, save_video)
from unscreen.vmatting import VMattingAgent


def main(cfg):
    print("Processing ", cfg['data'])
    src_img_dir = cfg["data"]["src_img_dir"]
    dst_img_dir = cfg["data"]["dst_img_dir"]
    os.makedirs(dst_img_dir, exist_ok=True)

    # read data
    st = time.time()
    frame_paths = sorted(glob(osp.join(src_img_dir, cfg["data"]["src_img_tmpl"])))
    if cfg["data"]["range"]:
        frame_paths = frame_paths[cfg["data"]["range"][0]:cfg["data"]["range"][1]]
    frame_list = parallel_read_img(frame_paths)
    h, w, _ = frame_list[0].shape
    numframes = len(frame_list)
    print('{} frames. Reading Data Done! {:.2f}s'.format(numframes, time.time() - st))

    # build agent
    segagent = SegAgent(**cfg['binseg'])
    stmagent = STMAgent(**cfg['stm'])
    trimapagent = TrimapAgent(**cfg['trimap'])
    vmatagent = VMattingAgent(**cfg['vmatting'])
    print('Agent built in {:.2f}s'.format(time.time() - st))

    tracking_flag = False
    alpha_pre = np.zeros((h, w), np.uint8)
    for fid in tqdm.tqdm(range(numframes)):
        # get segmask using video segmentation
        if tracking_flag:
            # run video segmentation
            segmask = alpha_pre.copy()
            segmask[segmask >= 128] = 255
            segmask = stmagent.forward(frame_list[fid-1:fid+1], segmask)[-1]
        else:
            # run image segmentation
            segmask = segagent.forward(frame_list[fid])
        save_segmask_path = osp.join(dst_img_dir, 'segmask_{:06d}.jpg'.format(fid))
        save_img(segmask, save_segmask_path)

        if not exist_foreground(segmask, cfg['fg_exist_thr']):
            # if no foreground
            fg = np.zeros_like(frame_list[fid])
            alpha = np.zeros_like(segmask)
        else:
            frame = frame_list[fid]

            alphaor = remove_invalid_objects(cfg, segmask.copy())
            # alphaor = dilate_mask(erode_mask(segmask, 3, 2), 3, 2)
            trimap = trimapagent.forward(alphaor.copy())
            alpha = vmatagent.forward(frame.copy(), alpha_pre.copy(), trimap.copy())
            bg = get_bg(alpha, frame)

            # calculate each bg image according to bg inpainting
            alpha_bin_255 = alpha.copy()
            alpha_bin_255[alpha_bin_255 >  128] = 255
            alpha_bin_255[alpha_bin_255 <= 128] = 0
            alpha_bin_255 = dilate_mask(alpha_bin_255, 3, 2)

            bgimg = np.stack((regionfill(bg[:, :, 0], alpha_bin_255), regionfill(bg[:, :, 1], alpha_bin_255),regionfill(bg[:, :, 2], alpha_bin_255)), axis=2)
            # bgimg = cv2.inpaint(bg.copy(), alpha_bin_255.copy(), 3, cv2.INPAINT_TELEA)
            bgimg = bgimg.astype(np.uint8)
            save_bg_path = osp.join(dst_img_dir, 'bg_{:06d}.jpg'.format(fid))
            save_img(bgimg, save_bg_path)

            alphabg_raw = (np.abs(frame.astype(np.float32) - bgimg.copy().astype(np.float32))).astype(np.uint8)  # calculate the unmask region
            alphabg = cv2.cvtColor(alphabg_raw, cv2.COLOR_BGR2GRAY)
            alphabg[alphabg > cfg["bg_mask"]["thr"]] = 255
            alphabg = np.clip((alphabg.astype(np.float)), 0, 255).astype(np.uint8)
            alphabg = dilate_mask(alphabg, 4, 2)

            # calculate fg image according to bg image
            alpha_ensm = alpha.copy() * (alphabg//255)
            alphaor = remove_invalid_objects(cfg, alpha_ensm.copy())
            trimap = trimapagent.forward(alphaor.copy())
            alpha = vmatagent.forward(frame.copy(), alpha_pre.copy(), trimap.copy())
            save_mask_path = osp.join(dst_img_dir, 'alphamask_{:06d}.jpg'.format(fid))
            save_img(alpha, save_mask_path)

            bgimg[alpha == 0] = frame[alpha == 0]
            fg = get_fg(frame, alpha, bgimg)
            save_fg_path = osp.join(dst_img_dir, 'fg_{:06d}.jpg'.format(fid))
            save_img(fg, save_fg_path)
        alpha_pre = alpha.copy()
        tracking_flag = exist_foreground(alpha, cfg['fg_exist_thr'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/bg.json')
    parser.add_argument('-vid', '--video_id', type=str, default="1")
    parser.add_argument('--range', type=str, default=None, help="eg. 400-700")
    args = parser.parse_args()

    data_root = '/new-pool/ayrao/matting/data/greenmat'
    cfg = json.load(open(args.cfg))
    cfg["data"] = {}
    cfg["data"]["range"] = args.range
    if args.range:
        cfg["data"]["range"] = [int(i) for i in args.range.split('-')]
    cfg["data"]["video_id"] = args.video_id
    cfg["data"]["src_img_dir"] = osp.join(data_root, 'src_img', cfg["data"]["video_id"])
    cfg["data"]["src_img_tmpl"] = '*.jpg'
    cfg["data"]["dst_img_dir"] = osp.join(data_root, 'test_bg_img', cfg["data"]["video_id"])
    cfg["data"]["dst_vid_dir"] = osp.join(data_root, 'video')
    main(cfg)
