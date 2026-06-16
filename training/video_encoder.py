import os 
import cv2
import torch
import random
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from training.cavp_modules import ResNet3dSlowOnly


class CAVP_Inference(nn.Module):

    def __init__(self, video_encode, embed_dim: int, ckpt_path):
        super().__init__()

        # Video Encoder:
        self.video_encode = video_encode
        assert self.video_encode == "Slowonly_pool"
        self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)    # Doesn't matter to set pretrained=None, since we will load CAVP weight outside.
        # Video Project & Pooling Head:
        self.video_project_head = nn.Linear(2048, embed_dim)
        self.video_pool = nn.MaxPool1d(kernel_size=16)

        assert ckpt_path is not None
        print("Initalize Stage1 CAVP Model")
        self.init_from_ckpt(ckpt_path)

        # Logit Scale:
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def init_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.load_state_dict(new_model, strict=False)
        print(f"Restored *video_encoder* with {len(missing)} missing and {len(unexpected)} unexpected keys")


    def first_init_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        project_head = {}
        pool = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            if new_key.startswith("video_encoder"):
                new_key = new_key.replace("video_encoder.","")
                new_model[new_key] = model[key]
            if new_key.startswith("video_project_head"):
                new_key = new_key.replace("video_project_head.","")
                project_head[new_key] = model[key]
            if new_key.startswith("video_pool"):
                new_key = new_key.replace("video_pool.","")
                pool[new_key] = model[key]
        missing, unexpected = self.video_encoder.load_state_dict(new_model, strict=False)
        print(f"Restored *video_encoder* with {len(missing)} missing and {len(unexpected)} unexpected keys")
        missing, unexpected = self.video_project_head.load_state_dict(project_head, strict=False)
        print(f"Restored *video_project_head* with {len(missing)} missing and {len(unexpected)} unexpected keys")
        missing, unexpected = self.video_pool.load_state_dict(pool, strict=False)
        print(f"Restored *video_pool* with {len(missing)} missing and {len(unexpected)} unexpected keys")


    def encode_video(self, video, pool=False, normalize=True):

        # Video: B x T x 3 x H x W
        assert self.video_encode == "Slowonly_pool"
        video = video.permute(0, 2, 1, 3, 4)
        video_feat = self.video_encoder(video)
        bs, c, t, _, _ = video_feat.shape
        video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
        video_feat = self.video_project_head(video_feat)
        
        # Pooling:
        if pool:
            video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)
        
        # Normalize:
        if normalize:
            video_feat = F.normalize(video_feat, dim=-1)

        return video_feat


    def forward(self, video):
        video_features = self.encode_video(video, pool=False, normalize=True)
        return video_features
        # return video_features, self.logit_scale.exp()
     


class TrainDataset(Dataset):
    def __init__(self, processed_data_dir, video_type='ori+crop'):
        split = "train"
        self.frame_length = 32
        self.video_type = video_type
        
        self.videocavp_dir = os.path.join(processed_data_dir, split, 'cavp_feat')
        self.video4fps_dir = os.path.join(processed_data_dir, split, 'vgg_video_4fps')
        self.crop_video_dir = os.path.join(processed_data_dir, split, 'crop_video_4fps')
        self.move_video_dir = os.path.join(processed_data_dir, split, 'move_video_4fps')
        
        vgg_list = os.listdir(self.video4fps_dir)
        crop_list = os.listdir(self.crop_video_dir)
        move_list = os.listdir(self.move_video_dir)

        if 'crop' == video_type:
            self.video_files = sorted(crop_list)[::3]
        elif 'vgg+crop' == video_type:
            self.video_files = vgg_list + sorted(crop_list)[::3]
        elif 'vgg+move' == video_type:
            self.video_files = vgg_list + move_list
        elif 'vgg+crop+move' == video_type:
            self.video_files = vgg_list + sorted(crop_list)[::3] + move_list
        elif 'vgg+crop3' == video_type:
            self.video_files = vgg_list + crop_list
        print('Video type: {}  Sample Num: {}'.format(video_type, len(self.video_files)))

        video_shape = (224,224)
        self.img_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):

        video_filename = self.video_files[idx]
        if video_filename[19] == 'n':
            video_path = os.path.join(self.video4fps_dir, video_filename)
        elif video_filename[19] == 'c':
            video_path = os.path.join(self.crop_video_dir, video_filename)
        else:
            video_path = os.path.join(self.move_video_dir, video_filename)
        feat_path = os.path.join(self.videocavp_dir, video_filename[:18] + '_cavp.npy')

        video_feat = torch.tensor(np.load(feat_path).astype(np.float32))
        # video_feat = video_feat[:32,:]
        if video_feat.size(0) < 32:
            pad_length = self.frame_length - video_feat.size(0)
            padded_video_feat = torch.cat([video_feat, torch.zeros(pad_length, video_feat.size(1))], dim=0)
            video_feat = padded_video_feat
        elif video_feat.size(0) > 32:
            video_feat = video_feat[:32,:]
        # include video and crop_video

        cap = cv2.VideoCapture(video_path)
        feat_batch_list = []
        first_frame = True
        while cap.isOpened():
            frames_exists, rgb = cap.read()

            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
            else:
                cap.release()
                break

        # cavp_input = torch.cat(feat_batch_list,0).unsqueeze(0)
        cavp_input = torch.cat(feat_batch_list,0)
        if cavp_input.size(0) < 32:
            pad_length = self.frame_length - cavp_input.size(0)
            padded_cavp_input = torch.cat([cavp_input, torch.zeros(pad_length, 3, 224, 224)], dim=0)
            cavp_input = padded_cavp_input
        elif cavp_input.size(0) > 32:
            cavp_input = cavp_input[:32,:]
        cavp_input = cavp_input.unsqueeze(0)

        return {"video_feat": video_feat,   # 32 * 512
                "cavp_input": cavp_input}    # 1 * 32 * 3 * 224 * 224
                
                
def collate_fn(data):
    video_feats = torch.stack([example["video_feat"] for example in data])
    cavp_inputs = torch.cat([example["cavp_input"] for example in data], dim=0)

    return {
        "video_feats": video_feats,  # bs * 32 * 512
        "cavp_inputs": cavp_inputs   # bs * 32 * 3 * 224 * 224
    }


class TrainDataset_v2(Dataset):
    def __init__(self, processed_data_dir, video_type='vgg+crop'):
        split = "train"
        self.frame_length = 32
        self.video_type = video_type
        self.type_ratio = 0.5
        
        self.videocavp_dir = os.path.join(processed_data_dir, split, 'cavp_feat')
        self.video4fps_dir = os.path.join(processed_data_dir, split, 'vgg_video_4fps')
        self.video_files = os.listdir(self.video4fps_dir)

        if video_type == 'vgg+crop+move':
            self.type_ratio = 1/3
        print('Video type: {}  Sample Num: {}'.format(video_type, len(self.video_files)))

        video_shape = (224,224)
        self.img_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.video_files)
    
    def get_video_feat(self, feat_path):

        video_feat = torch.tensor(np.load(feat_path).astype(np.float32))
        # video_feat = video_feat[:32,:]
        if video_feat.size(0) < 32:
            pad_length = self.frame_length - video_feat.size(0)
            padded_video_feat = torch.cat([video_feat, torch.zeros(pad_length, video_feat.size(1))], dim=0)
            video_feat = padded_video_feat
        elif video_feat.size(0) > 32:
            video_feat = video_feat[:32,:]

        return video_feat
        
    def crop_cavp_input(self, video_path):

        cap = cv2.VideoCapture(video_path)
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if abs(total_frame) < 1e-9:
            raise Exception("the total_frame is zero, please check the video")
        
        random_number = random.random()
        if random_number < self.type_ratio:
            crop = True
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 计算裁剪区域的大小
            x_ratio, y_ratio = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
            crop_width = int(width * x_ratio)
            crop_height = int(height * y_ratio)
            # 随机选择裁剪区域的左上角坐标
            x = random.randint(0, int(width - crop_width))
            y = random.randint(0, int(height - crop_height))
        else:
            crop = False

        feat_batch_list = []
        first_frame = True
        while cap.isOpened():
            frames_exists, rgb = cap.read()

            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                if crop:
                    rgb=rgb[y:y+crop_height, x:x+crop_width]
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
            else:
                cap.release()
                break

        # cavp_input = torch.cat(feat_batch_list,0).unsqueeze(0)
        cavp_input = torch.cat(feat_batch_list,0)
        if cavp_input.size(0) < 32:
            pad_length = self.frame_length - cavp_input.size(0)
            padded_cavp_input = torch.cat([cavp_input, torch.zeros(pad_length, 3, 224, 224)], dim=0)
            cavp_input = padded_cavp_input
        elif cavp_input.size(0) > 32:
            cavp_input = cavp_input[:32,:]
        cavp_input = cavp_input.unsqueeze(0)

        return cavp_input
    
    def move_cavp_input(self, video_path):

        cap = cv2.VideoCapture(video_path)
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if abs(total_frame) < 1e-9:
            raise Exception("the total_frame is zero, please check the video")
        
        random_number = random.random()
        if random_number < self.type_ratio:
            move = True
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 设置输出视频的尺寸大小
            x_ratio, y_ratio = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
            move_width = int(width * x_ratio)
            move_height = int(height * y_ratio)

            start = random.choice([True, False])
            if width > height:
                x = 0 if start else width - move_width
                y = int((height - move_height) // 2)
                dist = int((width - move_width) // total_frame)
            else:
                x = int((width - move_width) // 2)
                y = 0 if start else height - move_height
                dist = int((height - move_height) // total_frame)
        else:
            move = False

        feat_batch_list = []
        first_frame = True
        while cap.isOpened():
            frames_exists, rgb = cap.read()

            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                if move:
                    rgb=rgb[y:y+move_height, x:x+move_width]
                            # 更新镜头移动的位置
                    if width > height:
                        if start:
                            x += dist
                        else:
                            x = x-dist
                    else:
                        if start:
                            y += dist
                        else:
                            y = y-dist
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
            else:
                cap.release()
                break

        # cavp_input = torch.cat(feat_batch_list,0).unsqueeze(0)
        cavp_input = torch.cat(feat_batch_list,0)
        if cavp_input.size(0) < 32:
            pad_length = self.frame_length - cavp_input.size(0)
            padded_cavp_input = torch.cat([cavp_input, torch.zeros(pad_length, 3, 224, 224)], dim=0)
            cavp_input = padded_cavp_input
        elif cavp_input.size(0) > 32:
            cavp_input = cavp_input[:32,:]
        cavp_input = cavp_input.unsqueeze(0)

        return cavp_input
    

    def cropmove_cavp_input(self, video_path):

        cap = cv2.VideoCapture(video_path)
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if abs(total_frame) < 1e-9:
            raise Exception("the total_frame is zero, please check the video")
        
        random_number = random.random()
        if random_number < self.type_ratio:
            crop = True
            move = False
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 计算裁剪区域的大小
            x_ratio, y_ratio = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
            crop_width = int(width * x_ratio)
            crop_height = int(height * y_ratio)
            # 随机选择裁剪区域的左上角坐标
            x = random.randint(0, int(width - crop_width))
            y = random.randint(0, int(height - crop_height))
        elif random_number < self.type_ratio * 2:
            move = True
            crop = False
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 设置输出视频的尺寸大小
            x_ratio, y_ratio = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
            move_width = int(width * x_ratio)
            move_height = int(height * y_ratio)

            start = random.choice([True, False])
            if width > height:
                x = 0 if start else width - move_width
                y = int((height - move_height) // 2)
                dist = int((width - move_width) // total_frame)
            else:
                x = int((width - move_width) // 2)
                y = 0 if start else height - move_height
                dist = int((height - move_height) // total_frame)
        else:
            crop = False
            move = False

        feat_batch_list = []
        first_frame = True
        while cap.isOpened():
            frames_exists, rgb = cap.read()

            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                if crop:
                    rgb=rgb[y:y+crop_height, x:x+crop_width]
                if move:
                    rgb=rgb[y:y+move_height, x:x+move_width]
                            # 更新镜头移动的位置
                    if width > height:
                        if start:
                            x += dist
                        else:
                            x = x-dist
                    else:
                        if start:
                            y += dist
                        else:
                            y = y-dist
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
            else:
                cap.release()
                break

        # cavp_input = torch.cat(feat_batch_list,0).unsqueeze(0)
        cavp_input = torch.cat(feat_batch_list,0)
        if cavp_input.size(0) < 32:
            pad_length = self.frame_length - cavp_input.size(0)
            padded_cavp_input = torch.cat([cavp_input, torch.zeros(pad_length, 3, 224, 224)], dim=0)
            cavp_input = padded_cavp_input
        elif cavp_input.size(0) > 32:
            cavp_input = cavp_input[:32,:]
        cavp_input = cavp_input.unsqueeze(0)

        return cavp_input


    def __getitem__(self, idx):
        if idx >= len(self.video_files):
            raise IndexError(f"Index {idx} is out of range for video_files list")

        while True:
            try:
                video_filename = self.video_files[idx]
                vgg_video_path = os.path.join(self.video4fps_dir, video_filename)
                feat_path = os.path.join(self.videocavp_dir, video_filename[:18] + '_cavp.npy')
                if self.video_type == 'vgg+crop':
                    cavp_input = self.crop_cavp_input(vgg_video_path)
                elif self.video_type == 'vgg+move':
                    cavp_input = self.move_cavp_input(vgg_video_path)
                else:
                    cavp_input = self.cropmove_cavp_input(vgg_video_path)
                video_feat = self.get_video_feat(feat_path)
                break
            except IndexError as e:
                break  
            except:
                idx = idx +1
       
        return {"video_feat": video_feat,   # 32 * 512
                "cavp_input": cavp_input}    # 1 * 32 * 3 * 224 * 224
                
                
def collate_fn_v2(data):
    video_feats = torch.stack([example["video_feat"] for example in data])
    cavp_inputs = torch.cat([example["cavp_input"] for example in data], dim=0)

    return {
        "video_feats": video_feats,  # bs * 32 * 512
        "cavp_inputs": cavp_inputs   # bs * 32 * 3 * 224 * 224
    }


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        
    def forward(self, input1, input2):
        cosine_sim = F.cosine_similarity(input1, input2, dim=1)
        loss = 1 - cosine_sim.mean()
        return loss