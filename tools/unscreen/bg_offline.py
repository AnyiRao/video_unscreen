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
from unscreen.utils import (adaptive_resize, build_score_map, dilate_mask,
                            get_bg, get_fg,
                            exist_foreground, parallel_read_img, regionfill,
                            remove_invalid_objects, save_img, save_video)
from unscreen.vmatting import VMattingAgent


def main(cfg):
    print("Processing ", cfg['data'])
    src_img_dir = cfg["data"]["src_img_dir"]
    dst_img_dir = cfg["data"]["dst_img_dir"]
    os.makedirs(dst_img_dir, exist_ok=True)

    save_bg_always_path = osp.join(dst_img_dir, 'always_bg.jpg')

    # read data
    st = time.time()
    frame_paths = sorted(glob(osp.join(src_img_dir, cfg["data"]["src_img_tmpl"])))
    if cfg["data"]["range"]:
        frame_paths = frame_paths[cfg["data"]["range"][0]:cfg["data"]["range"][1]]
    frame_list = parallel_read_img(frame_paths)
    mask_list, bg_list = [], []
    h, w, _ = frame_list[0].shape
    numframes = len(frame_list)
    print('{} frames. Reading Data Done! {:.2f}s'.format(numframes, time.time()-st))

    # build agent
    st = time.time()
    segagent = SegAgent(**cfg['binseg'])
    stmagent = STMAgent(**cfg['stm'])
    trimapagent = TrimapAgent(**cfg['trimap'])
    vmatagent = VMattingAgent(**cfg['vmatting'])
    print('Building Agents Done! {:.2f}s'.format(time.time()-st))

    # video segmentation and calculate each bg image with matting
    if 0:
        print("get segmask using video segmentation")
        tracking_flag = False
        alpha_pre = np.zeros((h, w), np.uint8)
        for fid in tqdm.tqdm(range(numframes)):
            if tracking_flag:
                # run video segmentation
                segmask = alpha_pre.copy()
                segmask[segmask >= 128] = 255
                segmask = stmagent.forward(frame_list[fid-1:fid+1], segmask)[-1]
            else:
                # run image segmentation
                segmask = segagent.forward(frame_list[fid])
            segmask_three_chanel = np.stack((segmask, segmask, segmask), axis=2)
            mask_list.append(segmask_three_chanel)
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

                bgimg = np.stack((regionfill(bg[:,:,0], alpha_bin_255), regionfill(bg[:,:,1], alpha_bin_255),regionfill(bg[:,:,2], alpha_bin_255)), axis=2)
                # bgimg = cv2.inpaint(bg.copy(), alpha_bin_255.copy(), 3, cv2.INPAINT_TELEA)
                bgimg = bgimg.astype(np.uint8)
                bg_list.append(bgimg)
                save_bg_path = osp.join(dst_img_dir, 'bg_{:06d}.jpg'.format(fid))
                save_img(bgimg, save_bg_path)

            alpha_pre = alpha.copy()
            tracking_flag = exist_foreground(alpha, cfg['fg_exist_thr'])

    # calculate an overall bg image according to video segmentation
    if 0:
        print("calculate overall bg image according to video segmentation and temporal constraints")
        if mask_list == []:
            framepaths = sorted(glob(osp.join(dst_img_dir, 'segmask_*.jpg')))
            mask_list = parallel_read_img(framepaths)
        assert len(frame_list) == len(mask_list)

        dst_frame = frame_list[0]
        h, w, _ = dst_frame.shape
        bg_image_raw = np.zeros((h, w, 3))
        bg_image_raw_cont = np.zeros((h, w, 3))

        for fid in tqdm.tqdm(range(numframes)):
            frame = frame_list[fid]
            src_mask = mask_list[fid]
            frame = adaptive_resize(frame, dst_frame)
            src_mask = adaptive_resize(src_mask, dst_frame)
            src_mask = dilate_mask(src_mask, 3, 2)
            bg_img_no_mask = frame * (np.ones_like(src_mask) - src_mask // 255).astype(np.float32)
            bg_image_raw_cont += (src_mask < 250).astype(np.float32)
            bg_image_raw += bg_img_no_mask.astype(np.float32)
        mask_always = ((bg_image_raw_cont <= 10)*255).astype(np.uint8)

        bg_image_raw_cont_copy = bg_image_raw_cont.copy()
        bg_image_raw_cont_copy[bg_image_raw_cont == 0] = 1
        bg_always = np.clip((bg_image_raw/bg_image_raw_cont_copy), 0, 255).astype(np.uint8)
        bg_always[mask_always == 255] = 0

        mask_always = cv2.cvtColor(mask_always, cv2.COLOR_BGR2GRAY)
        mask_always_dilated = dilate_mask(mask_always, 3, 2)
        bg_img = cv2.inpaint(bg_always, mask_always_dilated, 3, cv2.INPAINT_TELEA)
        save_img(bg_img, save_bg_always_path)

    # calculate fg image according to bg image
    if 1:
        if mask_list == []:
            framepaths = sorted(glob(osp.join(dst_img_dir, 'segmask_*.jpg')))
            mask_list = parallel_read_img(framepaths)
        if bg_list == []:
            framepaths = sorted(glob(osp.join(dst_img_dir, 'bg_*.jpg')))
            bg_list = parallel_read_img(framepaths)
        assert len(frame_list) == len(mask_list) and len(frame_list) == len(bg_list)
        bgimg_always = cv2.imread(save_bg_always_path)

        print("calculate fg image according to bg image")
        alpha_pre = None
        for fid in tqdm.tqdm(range(numframes)):
            frame = frame_list[fid]
            alpha = mask_list[fid]
            bgimg = bg_list[fid]

            beta = cfg["bg_mask"]["fusion_weight"]
            bgimg = ((bgimg.copy().astype(np.float32) * beta + (1-beta) * bgimg_always.copy().astype(np.float32))).astype(np.uint8)
            alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

            alphabg_raw = (np.abs(frame.astype(np.float32) - bgimg.copy().astype(np.float32))).astype(np.uint8)  # calculate the unmask region
            alphabg = cv2.cvtColor(alphabg_raw, cv2.COLOR_BGR2GRAY)
            alphabg[alphabg > cfg["bg_mask"]["thr"]] = 255
            alphabg = np.clip((alphabg.astype(np.float32)), 0, 255).astype(np.uint8)
            alphabg = dilate_mask(alphabg, 4, 2)

            alpha = alpha.copy() * (alphabg//255)

            if alpha_pre is None:
                alpha_pre = alpha

            alphaor = remove_invalid_objects(cfg, alpha.copy())
            trimap = trimapagent.forward(alphaor.copy())
            alpha = vmatagent.forward(frame.copy(), alpha_pre.copy(), trimap.copy())
            save_mask_path = osp.join(dst_img_dir, 'alphamask_{:06d}.jpg'.format(fid))
            save_img(alpha, save_mask_path)

            bgimg[alpha == 0] = frame[alpha == 0]
            fg = get_fg(frame, alpha, bgimg)

            save_fg_path = osp.join(dst_img_dir, 'fg_{:06d}.jpg'.format(fid))
            save_img(fg, save_fg_path)
            alpha_pre = alpha.copy()
        save_video(cfg, 'fg')


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
    cfg["data"]["dst_img_dir"] = osp.join(data_root, 'test_img', cfg["data"]["video_id"])
    cfg["data"]["dst_vid_dir"] = osp.join(data_root, 'video')
    main(cfg)
