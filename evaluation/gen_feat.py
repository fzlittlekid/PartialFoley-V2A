import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
device = torch.device("cuda")  # Set Device:

import argparse
import numpy as np
from tqdm import tqdm

from demo_util import Extract_CAVP_Features 


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, default='')
    parser.add_argument('--video_type', type=str, default='vgg', help="vgg or crop or move")
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--save_video_dir', type=str, default='')
    parser.add_argument('--tmp_path', type=str, default='')

    parser.add_argument('--eval_method', type=str, default='Ours', help="Ours or Diff-Foley")
    parser.add_argument('--cavp_ckpt_path', type=str, default='')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    video_list = []
    with open(args.file_path, "r") as f:
        video_list = f.readlines()
        video_list = list(map(lambda x: x.strip(), video_list))
    video_list = video_list
    # print("[5368:]")    

    video_type = args.video_type
    video_dir = os.path.join(args.save_video_dir, 'test', '{}_video'.format(video_type))
    feat_save_dir = os.path.join(args.exp_dir, 'feat', video_type)
    os.makedirs(feat_save_dir, exist_ok=True)

    eval_method = args.eval_method
    cavp_ckpt_path = args.cavp_ckpt_path
    print('eval_method: ', eval_method)  # Ours or Diff-Foley

    if eval_method == 'Ours':
        extract_cavp = Extract_CAVP_Features(method='Ours', ckpt_path=cavp_ckpt_path, tmp_path=args.tmp_path, device=device)
    elif eval_method == 'Diff-Foley':
        cavp_config_path = "./diff_foley/config/Stage1_CAVP.yaml"              #  CAVP Config
        extract_cavp = Extract_CAVP_Features(method='Diff-Foley', ckpt_path=cavp_ckpt_path, tmp_path=args.tmp_path, device=device, diff_config_path=cavp_config_path)
    else:
        raise Exception("method: Ours or Diff-Foley")

    print('|| video_type: ', video_type)
    print('test_video_dir: ', video_dir)
    print('feat_save_dir: ', feat_save_dir)

    extension = '.mp4' if video_type == 'vgg' else '_{}0.mp4'.format(video_type)
    if video_type == 'move':
        extension = '_out0.mp4'
    for video_name in tqdm(video_list):
        video_path = os.path.join(video_dir, video_name + extension)
        cavp_feats = extract_cavp(video_path, 0, 10, test="eval")
        np.save(os.path.join(feat_save_dir, video_name + '_feat.npy'), cavp_feats)


if __name__ == "__main__":
    main()
