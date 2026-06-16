import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
device = torch.device("cuda")  # Set Device:

import random
import argparse
import numpy as np
import soundfile as sf

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from demo_util import load_model_from_config, inverse_op


def seed_everything(seed):
    # import random, os
    # import numpy as np
    # import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class TestDataset(Dataset):
    def __init__(self, file_path, feat_save_dir, video_type):
        self.video_type = video_type

        with open(file_path, "r") as f:
            video_list = f.readlines()
            video_list = list(map(lambda x: x.strip(), video_list))

        self.video_list = video_list   

        self.feat_save_dir = feat_save_dir
        print('feat_save_dir: ', self.feat_save_dir)
        
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        cavp_feat_path = os.path.join(self.feat_save_dir, self.video_list[idx] + '_feat.npy')
        cavp_feat = torch.tensor(np.load(cavp_feat_path).astype(np.float32))
        return cavp_feat, self.video_list[idx]
    


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, default='')
    parser.add_argument('--video_type', type=str, default='vgg', help="vgg or crop or move")
    parser.add_argument('--exp_dir', type=str, default='')
    
    parser.add_argument('--ldm_ckpt_path', type=str, default='')
    parser.add_argument('--sample_num', type=int, default=10)

    args = parser.parse_args()
    return args


def main():
    seed_everything(21)
    args = parse_args()

    video_type = args.video_type  # 'vgg' or 'crop'  or 'move'
    feat_save_dir = os.path.join(args.exp_dir, 'feat', video_type)

    save_mel_path = os.path.join(args.exp_dir, 'mel', video_type)
    os.makedirs(save_mel_path, exist_ok=True)
    print(save_mel_path)
    # gen wav
    save_wav_path = os.path.join(args.exp_dir, 'wav', video_type)
    os.makedirs(save_wav_path, exist_ok=True)
    print(save_wav_path)

    test_dataset = TestDataset(file_path=args.file_path, feat_save_dir=feat_save_dir, video_type=video_type)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ldm_config_path = "./diff_foley/config/Stage2_LDM.yaml"
    config = OmegaConf.load(ldm_config_path)
    latent_diffusion_model = load_model_from_config(config, args.ldm_ckpt_path)


    # 3. Diff-Foley Generation
    cfg_scale = 4.5
    steps = 25
    sampler = "DPM_Solver"
    truncate_len = 32
    sample_num = args.sample_num

    # 4. Iterate over the test data and generate audio for each video
    for j, (cavp_feats, file_name) in enumerate(tqdm(test_dataloader)):

        video_feat = cavp_feats[0].unsqueeze(0).repeat(sample_num, 1, 1).to(device)
        feat_len = video_feat.shape[1]
        window_num = feat_len // truncate_len    # window_num == 1 

        audio_list = []
        for i in range(window_num):
            start, end = i * truncate_len, (i+1) * truncate_len
            embed_cond_feat = latent_diffusion_model.get_learned_conditioning(video_feat[:, start:end])
            uncond_cond = torch.zeros(embed_cond_feat.shape).to(device)

            audio_samples, _ = latent_diffusion_model.sample_log_diff_sampler(embed_cond_feat, batch_size=sample_num, sampler_name=sampler, ddim_steps=steps, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=uncond_cond)

            audio_samples = latent_diffusion_model.decode_first_stage(audio_samples)
            audio_samples = audio_samples[:, 0, :, :].detach().cpu().numpy()

            sample_list = []
            for j in range(audio_samples.shape[0]):
                # gen mel
                mel_spec = audio_samples[j]     # [128,512]
                np.save(os.path.join(save_mel_path, "{}_{}_mel.npy".format(file_name[0], j)), mel_spec)
                # gen wav
                sample = inverse_op(audio_samples[j])
                sample_list.append(sample)
            audio_list.append(sample_list)

        for i in range(sample_num):      # gen wav
            current_audio_list = []
            for k in range(window_num):
                current_audio_list.append(audio_list[k][i])
            if current_audio_list:
                current_audio = np.concatenate(current_audio_list,0)
                sf.write(os.path.join(save_wav_path, "{}_{}.wav").format(file_name[0], i), current_audio, 16000)
            else:
                print("eval_wav_mel_err:", file_name[0])


if __name__ == "__main__":
    main()