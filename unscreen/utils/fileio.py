"""some functions for file IO."""
import os
import os.path as osp
from multiprocessing import Pool
from datetime import datetime
from glob import glob
import cv2
import mmcv


def read_txt_list(txt_fn):
    with open(txt_fn, "r") as f:
        txt_list = f.read().splitlines()
    return txt_list


def write_txt_list(txt_fn, txt_list):
    with open(txt_fn, "w") as f:
        for item in txt_list:
            f.write("{}\n".format(item))


def return_date():
    return datetime.now().strftime('%m%d')


def return_time():
    return datetime.now().strftime('%m%d') + "_" + datetime.now().strftime('%H%M')


def run_read_img(i, framepath):
    return cv2.imread(framepath)


def parallel_read_img(framepaths):
    pool = Pool(48)
    framelist = pool.starmap(run_read_img, [(i, framepath) for i, framepath in enumerate(framepaths)])
    return framelist


def save_video(cfg, mode):
    vid = cfg["data"]["video_id"]
    os.makedirs(cfg['data']['dst_vid_dir'], exist_ok=True)
    mmcv.frames2video(cfg['data']['dst_img_dir'] ,
                      osp.join(cfg['data']['dst_vid_dir'], '{}_{}.mp4'.format(vid, mode)),
                      filename_tmpl=mode+"_{:06d}.jpg")
    print("saving the {} of video {}".format(mode, vid))


def save_img(img, save_path, downsacle=1):
    """save an image, the image would be down sampled if needed

    Args:
        img (np.array<uint8>): image to save
        save_path (str): the path to save the image
        downsacle (int): the scale to down sample the image
    """
    assert isinstance(downsacle, int)
    if downsacle != 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsacle, h // downsacle))
    cv2.imwrite(save_path, img)


def delete_mode(cfg, mode):
    dellist = sorted(glob(osp.join(cfg["data"]["save_img_dir"], '{}_*.jpg'.format(mode))))
    [os.remove(i) for i in dellist]
