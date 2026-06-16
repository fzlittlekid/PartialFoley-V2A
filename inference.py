import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
device = torch.device("cuda")  # Set Device:

import random
import numpy as np
import soundfile as sf

from tqdm import tqdm
from omegaconf import OmegaConf

from evaluation.demo_util import load_model_from_config, inverse_op
from evaluation.demo_util import Extract_CAVP_Features

import librosa
import librosa.display
import matplotlib.pyplot as plt


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
seed_everything(21)



# 0. Sample and save path
# Sample1:
video_path = ""
tmp_path = "" 
save_path = ""
os.makedirs(save_path, exist_ok=True)


# 1. Loading Stage1 CAVP Model:

method = "Ours"  # Ours or Diff-Foley
print('eval_method: ', method)

if method == 'Ours':
    cavp_ckpt_path =""
    extract_cavp = Extract_CAVP_Features(method='Ours', ckpt_path=cavp_ckpt_path, tmp_path=tmp_path, device=device)
elif method == 'Diff-Foley':
    cavp_config_path = "./diff_foley/config/Stage1_CAVP.yaml"              #  CAVP Config
    cavp_ckpt_path = ""      #  CAVP Ckpt
    extract_cavp = Extract_CAVP_Features(method='Diff-Foley', ckpt_path=cavp_ckpt_path, tmp_path=tmp_path, device=device, diff_config_path=cavp_config_path)
else:
    raise Exception("method: Ours or Diff-Foley")
extract_cavp = extract_cavp.to(device)


# 2. Loading Stage2 LDM Model:

# LDM Config:
ldm_config_path = "./diff_foley/config/Stage2_LDM.yaml"
ldm_ckpt_path = ""
config = OmegaConf.load(ldm_config_path)
# Loading LDM:
latent_diffusion_model = load_model_from_config(config, ldm_ckpt_path)

# Whether use Double Guidance:
use_double_guidance = False
if use_double_guidance:
    classifier_config_path = "./diff_foley/config/Double_Guidance_Classifier.yaml"
    classifier_ckpt_path = ""
    classifier_config = OmegaConf.load(classifier_config_path)
    classifier = load_model_from_config(classifier_config, classifier_ckpt_path)


# 3. Extract Video CAVP Features & New Video Path:
start_second = 0              # Video start second
truncate_second = 3         # Video end = start_second + truncate_second
cavp_feats, new_video_path = extract_cavp(video_path, start_second, truncate_second, test="demo")

# 4. Diff-Foley Generation:
# Inference Param:
cfg_scale = 4.5      # Classifier-Free Guidance Scale
cg_scale = 50        # Classifier Guidance Scale
steps = 25                # Inference Steps
sampler = "DPM_Solver"    # or "DDIM" or "PLMS"  

# Video CAVP Features:
sample_num = 4
video_feat = torch.from_numpy(cavp_feats).unsqueeze(0).repeat(sample_num, 1, 1).to(device)

# Truncate the Video Cond:
feat_len = video_feat.shape[1]
truncate_len = 12
window_num = feat_len // truncate_len


audio_list = []     # [sample_list1, sample_list2, sample_list3 ....]
for i in tqdm(range(window_num), desc="Window:"):
    start, end = i * truncate_len, (i+1) * truncate_len
    
    # 1). Get Video Condition Embed:
    embed_cond_feat = latent_diffusion_model.get_learned_conditioning(video_feat[:, start:end])  

    # 2). CFG unconditional Embedding:
    uncond_cond = torch.zeros(embed_cond_feat.shape).to(device)
    
    # 3). Diffusion Sampling:
    print("Using Double Guidance: {}".format(use_double_guidance))
    if use_double_guidance:
        audio_samples, _ = latent_diffusion_model.sample_log_with_classifier_diff_sampler(embed_cond_feat, origin_cond=video_feat, batch_size=video_feat.shape[0], sampler_name=sampler, ddim_steps=steps, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=uncond_cond,classifier=classifier, classifier_guide_scale=cg_scale)  # Double Guidance
    else:
        audio_samples, _ = latent_diffusion_model.sample_log_diff_sampler(embed_cond_feat, batch_size=sample_num, sampler_name=sampler, ddim_steps=steps, unconditional_guidance_scale=cfg_scale,unconditional_conditioning=uncond_cond)           #  Classifier-Free Guidance
    
    # 4). Decode Latent:
    audio_samples = latent_diffusion_model.decode_first_stage(audio_samples)    
    audio_samples = audio_samples[:, 0, :, :].detach().cpu().numpy()                           

    # 5). Spectrogram -> Audio:  (Griffin-Lim Algorithm)
    sample_list = []        #    [sample1, sample2, ....]
    for k in tqdm(range(audio_samples.shape[0]), desc="current samples:"):
        sample = inverse_op(audio_samples[k])
        sample_list.append(sample)
    audio_list.append(sample_list)

# Save Samples:
path_list = []
for i in range(sample_num):      # sample_num
    current_audio_list = []
    for k in range(window_num):
        current_audio_list.append(audio_list[k][i])
    current_audio = np.concatenate(current_audio_list,0)
    print(current_audio.shape)
    sf.write(os.path.join(save_path, "sample_{}_diff.wav").format(i), current_audio, 16000)
    path_list.append(os.path.join(save_path, "sample_{}_diff.wav").format(i))
print("Gen Success !!")

# Concat The Video and Sound:
import subprocess
src_video_path = new_video_path
for i in range(sample_num):
    gen_audio_path = path_list[i]
    out_path = os.path.join(save_path, "output_{}.mp4".format(i))
    cmd = ["ffmpeg" ,"-i" ,src_video_path,"-i" , gen_audio_path ,"-c:v" ,"copy" ,"-c:a" ,"aac" ,"-strict" ,"experimental", out_path]
    subprocess.check_call(cmd)
print("Gen Success !!")
