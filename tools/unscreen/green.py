import argparse
import json
import os
import os.path as osp
import pdb
import time
from glob import glob

import numpy as np
import tqdm
from unscreen.binseg import SegAgent
from unscreen.colorfiltering import ColorFilteringAgent
from unscreen.trimap import TrimapAgent
from unscreen.utils import (exist_foreground, color_correct,
                            get_fg, parallel_read_img,
                            remove_invalid_objects,
                            save_img, save_video)
from unscreen.vmatting import VMattingAgent


def print_statistic(runtime, tracking_count, numframes):
    """print some statistic information."""
    print(f'{tracking_count} / {numframes} use tracking')
    print('-' * 10 + 'runtime' + '-' * 10)
    for key, value in runtime.items():
        print(f'{key:>16s}: {value / numframes:.2f}s')
    print('-' * 10 + '-------' + '-' * 10)
    print('\n')


def main(cfg):
    src_img_dir = cfg['data']['src_img_dir']
    dst_img_dir = cfg["data"]["dst_img_dir"]
    dst_vid_dir = cfg['data']['dst_vid_dir']
    os.makedirs(src_img_dir, exist_ok=True)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_vid_dir, exist_ok=True)

    # build agent
    st = time.time()
    segagent = SegAgent(**cfg['binseg'])
    trimapagent = TrimapAgent(**cfg['trimap'])
    vmatagent = VMattingAgent(**cfg['vmatting'])
    cfagent = ColorFilteringAgent(**cfg['colorfiltering'])
    print(f'Building Agents Done! {time.time() - st:.2f}s')

    # read data
    st = time.time()
    frame_paths = sorted(glob(osp.join(src_img_dir, cfg["data"]["src_img_tmpl"])))
    if cfg["data"]["range"]:
        frame_paths = frame_paths[cfg["data"]["range"][0]:cfg["data"]["range"][1]]
    frame_list = parallel_read_img(frame_paths)
    numframes = len(frame_list)
    h, w, _ = frame_list[0].shape
    print('{} frames. Reading Data Done! {:.2f}s'.format(numframes, time.time()-st))

    # statistic information
    tracking_count = 0
    runtime = {
        'seg': 0,
        'color_filter': 0,
        'object_removal': 0,
        'matting': 0,
        'color_correct': 0,
        'getfg': 0
    }
    cfagent.reset_gmms()  # reset gmm models in cfagent
    tracking_flag = False
    alpha_pre = np.zeros((h, w), np.uint8)
    for fid in tqdm.tqdm(range(numframes)):
        # frame -> segmentation -> color filtering -> object removal -> matting -> forground
        frame = frame_list[fid]
        # 1. segmentation
        if tracking_flag:
            # video segmentation: use the result of last frame
            segmask = alpha_pre.copy()
            tracking_count += 1
        else:
            # image segmentation
            segmask = segagent.forward(frame.copy())
        # if no foreground
        if not exist_foreground(segmask, cfg['fg_exist_thr']):
            alpha = np.zeros_like(segmask)
            fg = np.zeros_like(frame)
            bgimg = frame
        else:
            st = time.time()
            if fid % cfg['colorfiltering_update_duration'] == 0:
                cf_iters = cfg['colorfiltering_train_iters']
            # new object appears
            elif (not tracking_flag) and exist_foreground(segmask, cfg['fg_exist_thr']):
                cf_iters = cfg['colorfiltering_train_iters']
            elif not cfagent.is_trained():
                cf_iters = cfg['colorfiltering_train_iters']
            else:
                cf_iters = 0

            # 2. color filtering
            alphacf, bgimg, _ = cfagent.forward(
                frame.copy(), segmask.copy(), iters=cf_iters)
            bg_color = bgimg[0, 0]
            runtime['color_filter'] += time.time() - st

            # 3. remove some invalid objects
            st = time.time()
            if tracking_flag:
                alphaor = remove_invalid_objects(cfg, alphacf.copy())
            else:
                alphaor = remove_invalid_objects(cfg, alphacf.copy(), segmask.copy())
            runtime['object_removal'] += time.time() - st

            # 4. matting
            st = time.time()
            trimap = trimapagent.forward(alphaor.copy(), frame.copy(), bg_color)
            alpha = vmatagent.forward(frame.copy(), alpha_pre.copy(), trimap.copy())
            runtime['matting'] += time.time() - st

            # 5. color correction
            st = time.time()
            alpha = color_correct(frame.copy(), alpha.copy(), bg_color.copy())
            runtime['color_correct'] += time.time() - st

            # 6. get fg
            st = time.time()
            bgimg[alpha < 128] = frame[alpha < 128]
            fg = get_fg(frame.copy(), alpha.copy(), bgimg.copy())
            runtime['getfg'] += time.time() - st

        save_mask_path = osp.join(dst_img_dir, 'alphamask_{:06d}.jpg'.format(fid))
        save_fg_path =   osp.join(dst_img_dir, 'fg_{:06d}.jpg'.format(fid))
        save_bg_path =   osp.join(dst_img_dir, 'bg_{:06d}.jpg'.format(fid))
        save_img(fg, save_fg_path)
        save_img(alpha, save_mask_path)
        save_img(bgimg, save_bg_path)

        # set tracking flag
        tracking_flag = exist_foreground(alpha, cfg['fg_exist_thr'])
        alpha_pre = alpha.copy()

    print_statistic(runtime, tracking_count, numframes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/green.json')
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
    cfg["data"]["src_img_tmpl"] = '*.*'
    cfg["data"]["dst_img_dir"] = osp.join(data_root, 'test_green_img', cfg["data"]["video_id"])
    cfg["data"]["dst_vid_dir"] = osp.join(data_root, 'video')
    main(cfg)
