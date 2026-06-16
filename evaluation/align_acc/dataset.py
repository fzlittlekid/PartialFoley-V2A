
import os
import torch
import numpy as np


class VGGSound_audio_video_spec_fullset_Dataset_Infer(torch.utils.data.Dataset):
    def __init__(self, feat_dir, eval_dataset_path, split, data_dir, feat_type='CAVP_feat', sr=22050, duration=10, truncate=220000, fps=21.5, debug_num=False, fix_frames=True):
        super().__init__()
        self.data_dir = data_dir

        if split == "train":
            self.split_path = os.path.join(self.data_dir, "Train")  # spec dir
            self.split = "Train"
        elif split == "valid" or split == 'test':
            self.split_path = os.path.join(self.data_dir, "Test")   # spec dir
            self.split = "Test"

        # Default params:
        self.sr = sr                # 22050
        self.duration = duration    # 10
        self.truncate = truncate    # 220000
        self.fps = fps
        self.fix_frames = fix_frames
        print("Fix Frames: {}".format(self.fix_frames))

        self.feat_dir = feat_dir   # cavp feat data path
        self.eval_datset_path = eval_dataset_path  # Generate Data Path

        self.data_list = os.listdir(self.eval_datset_path)
        self.audio_name_list = list(map(lambda x: "_".join(x.split("_")[:-2]), self.data_list))

        print('Split: {}  Sample Num: {}'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):

        audio_name = self.audio_name_list[idx]             # 
       
        video_feat_path = os.path.join(self.feat_dir, audio_name + "_feat.npy")
        video_feat = np.load(video_feat_path).astype(np.float32)
       
        start_frame = 0
        # truncate_frame = int(self.fps * self.truncate / self.sr) 
        truncate_frame = 32
        if video_feat.shape[0] < start_frame + truncate_frame:
            repeat_num = int((start_frame + truncate_frame) // video_feat.shape[0]) + 1
            video_feat = np.tile(video_feat, (repeat_num,1))
            video_feat = video_feat[start_frame: start_frame + truncate_frame]
        else:
            video_feat = video_feat[start_frame: start_frame + truncate_frame]
        # video_feat = video_feat[start_frame:start_frame + truncate_frame]

        spec_path = os.path.join(self.eval_datset_path, self.data_list[idx])
        spec = np.load(spec_path).astype(np.float32)
        spec_truncate = 512
        if spec.shape[-1] < spec_truncate:
            repeat_num = int(spec_truncate // spec.shape[-1]) + 1
            spec = np.tile(spec, repeat_num)               # repeat 2 
            spec = spec[:, :spec_truncate]
        else:
            spec = spec[:, :spec_truncate]
        spec = spec[:, :spec_truncate]

        data_dict = {}
        # data_dict['spec'] = audio
        data_dict['audio_name'] = audio_name
        data_dict['video_feat'] = video_feat
        data_dict["spec"] = spec[None].repeat(3, axis=0)
        data_dict["labels"] = torch.tensor(1)
        return data_dict



class VGGSound_audio_video_spec_fullset_Dataset_Train_Infer(VGGSound_audio_video_spec_fullset_Dataset_Infer):
    def __init__(self, dataset_cfg):
        super().__init__(split='train', **dataset_cfg)

class VGGSound_audio_video_spec_fullset_Dataset_Valid_Infer(VGGSound_audio_video_spec_fullset_Dataset_Infer):
    def __init__(self, dataset_cfg):
        super().__init__(split='valid', **dataset_cfg)

class VGGSound_audio_video_spec_fullset_Dataset_Test_Infer(VGGSound_audio_video_spec_fullset_Dataset_Infer):
    def __init__(self, dataset_cfg):
        super().__init__(split='test', **dataset_cfg)