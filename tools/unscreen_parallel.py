import argparse
from tqdm import tqdm
import os
import os.path as osp
import pdb
import sys
sys.path.append(".")
from unscreen.utils import read_txt_list

parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str, default="green")
parser.add_argument('--gpuid', type=str, default="1")
parser.add_argument('--range', type=str, default=None, help="eg. 0-40")
args = parser.parse_args()

data_root = "/new-pool/ayrao/matting/data/greenmat"
vid_list = read_txt_list(osp.join(data_root, 'meta/vid_list.txt'))

start, end = args.range.split("-")
vid_list = vid_list[int(start): int(end)]

for vid in tqdm(vid_list):
    cmd = "bash tools/unscreen.sh {} {} {}".format(args.script, vid, args.gpuid)
    os.system(cmd)
print('\nVideos Done: ', len(vid_list))
