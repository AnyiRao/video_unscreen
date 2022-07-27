import argparse
import json
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

sys.path.append(".")
from unscreen.utils import (get_connectivity, get_gradient_error, get_miou,
                            get_mse, get_sad, read_txt_list, write_txt_list)


def print_metrics(cfg, miou_dict, sad_dict, mse_dict, grad_dict, conn_dict):
    miou_list = []
    sad_list = []
    mse_list, grad_list, conn_list = [], [], []
    save_list = []
    print('-' * 50)
    for key, vid_mious in miou_dict.items():
        miou_result = np.array(miou_dict[key]).mean()
        sad_result = np.array(sad_dict[key]).mean()
        mse_result = np.array(mse_dict[key]).mean()
        grad_result = np.array(grad_dict[key]).mean()
        conn_result = np.array(conn_dict[key]).mean()
        result_item_save = "{} MIOU: {:.06g} SAD: {:.06g} MSE: {:.06g} GRAD: {:.06g} CONN: {:.06g}'".format(
            key, miou_result, sad_result, mse_result, grad_result, conn_result)
        print(result_item_save)
        save_list.append(result_item_save)
        miou_list.append(miou_result)
        sad_list.append(sad_result)
        mse_list.append(mse_result)
        grad_list.append(grad_result)
        conn_list.append(conn_result)
    print('-' * 50)
    result_item_save = "{} MIOU: {:.06g} SAD: {:.06g} MSE: {:.06g} GRAD: {:.06g} CONN: {:.06g}'".format(
        'ALL',
        np.array(miou_list).mean(),
        np.array(sad_list).mean(),
        np.array(mse_list).mean(),
        np.array(grad_list).mean(),
        np.array(conn_list).mean()
        )
    print(result_item_save)
    save_list.append(result_item_save)
    print('-' * 50)
    txt_fn = cfg["data"]["save_data_fn"]
    write_txt_list(txt_fn, save_list)


def evaluate_one(pairs):
    gt_path, pred_path = pairs
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    miou = get_miou(gt_img, pred_img)
    sad = get_sad(gt_img, pred_img)
    mse = get_mse(gt_img, pred_img)
    grad = get_gradient_error(gt_img, pred_img)
    conn = get_connectivity(gt_img, pred_img)
    return (miou, sad, mse, grad, conn)


def run(cfg):
    gt_data_dir = cfg["data"]["gt_data_dir"]
    pred_data_dir = cfg["data"]["pred_data_dir"]

    miou_dict = {}
    sad_dict = {}
    mse_dict, grad_dict, conn_dict = {}, {}, {}
    vid_list = read_txt_list(cfg["data"]["meta_fn"])
    for vid in tqdm(vid_list):
        gt_framepaths = sorted(glob(osp.join(gt_data_dir, vid, cfg["data"]["gt_data_tmpl"])))
        pred_framepaths = sorted(glob(osp.join(pred_data_dir, vid, cfg["data"]["pred_data_tmpl"])))
        # get metrics
        miou_dict[vid], sad_dict[vid] = [], []
        mse_dict[vid], grad_dict[vid], conn_dict[vid] = [], [], []
        pairs = []
        for i in range(len(gt_framepaths)):
            gt_path = gt_framepaths[i]
            pred_path = pred_framepaths[i]
            pairs.append((gt_path, pred_path))
        results = mmcv.track_parallel_progress(evaluate_one, pairs, nproc=24)

        for miou, sad, mse, grad, conn in results:
            miou_dict[vid].append(miou)
            sad_dict[vid].append(sad)
            mse_dict[vid].append(mse)
            grad_dict[vid].append(grad)
            conn_dict[vid].append(conn)
    print_metrics(cfg, miou_dict, sad_dict, mse_dict, grad_dict, conn_dict)


if __name__ == '__main__':
    data_root = '/new-pool/ayrao/matting/data/greenmat'
    # data_root = '/new-pool/ayrao/matting/data/adobemat'
    exp_name = "test_green"
    cfg = {}
    cfg["data"] = {}
    cfg["data"]["range"] = None
    cfg["data"]["meta_fn"] = osp.join(data_root, 'meta/vid_list2.txt')
    cfg["data"]["gt_data_dir"] = osp.join(data_root, 'alpha_img')
    cfg["data"]["gt_data_tmpl"] = '*.*'
    cfg["data"]["pred_data_dir"] = osp.join(data_root, f'{exp_name}_img')
    cfg["data"]["pred_data_tmpl"] = 'alphamask_*.*'
    cfg["data"]["save_data_fn"] = osp.join(data_root, f'results/{exp_name}.txt')
    run(cfg)
