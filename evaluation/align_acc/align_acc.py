import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
device = torch.device("cuda")

import numpy as np
from tqdm import tqdm
import datetime
from omegaconf import OmegaConf
from evaluation.demo_util import instantiate_from_config, load_model_from_config

# distributed:
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def collate_fn(batch):

    audio_names = [data_dict['audio_name'] for data_dict in batch if data_dict is not None]
    video_feats = [data_dict['video_feat'] for data_dict in batch if data_dict is not None]
    specs = [data_dict['spec'] for data_dict in batch if data_dict is not None]
    labels = torch.tensor([data_dict['labels'] for data_dict in batch if data_dict is not None])
    
    # 将视频特征和光谱数据转换为张量
    video_feats = [torch.from_numpy(feat) for feat in video_feats]
    specs = [torch.from_numpy(spec) for spec in specs]
    
    # 将视频特征和光谱数据堆叠成批处理张量
    video_feats = torch.stack(video_feats, dim=0)
    specs = torch.stack(specs, dim=0)
    
    # 创建批处理字典
    batch_dict = {
        'audio_name': audio_names,
        'video_feat': video_feats,
        'spec': specs,
        'labels': labels
    }
    
    return batch_dict


def load_model_and_dataloaders(cfg, ckpt, device, is_ddp=False, eval_method="Ours"):
    
    # load model:
    model = load_model_from_config(cfg, ckpt)
    model = model.to(device)
    # get data:
    if eval_method == "Ours":
        print("Evaluating Ours Method: ========> ")
        data = instantiate_from_config(cfg.data_eval_metric)
    data.prepare_data()
    data.setup()

    if is_ddp:
        print(is_ddp)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        sampler = DistributedSampler(data.datasets['validation'], dist.get_world_size(), dist.get_rank(), shuffle=False)
        num_workers = 0
    else:
        sampler = None
        num_workers = 8
    
    dataloader = DataLoader(data.datasets['validation'],sampler=sampler, collate_fn=collate_fn, batch_size=cfg.data_eval_metric.batch_size, num_workers=num_workers, pin_memory=True, drop_last=False) 

    return dataloader, model


def model_inference(model, batch, guidance_scale=1.0):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    print("Guidance Scale: {}".format(guidance_scale))
    model.eval()

    # labels = batch["labels"].to(model.device)

    bs = batch['spec'].shape[0]
    labels = torch.ones(bs).to(model.device)

    with torch.no_grad():
        spec = batch["spec"].to(model.device)
        video_feat = batch["video_feat"].to(model.device)
        encode_spec = model.encode_spec_z(spec)
        encode_cond = model.cond_model(video_feat)
        t = torch.tensor(0).reshape(1,).repeat(spec.shape[0]).to(model.device).long()   # Constant
        prob_logits = model.model(encode_spec, context=encode_cond, timesteps=t)
        predicted = torch.round(prob_logits)
        correct_num = ((predicted == labels.float().unsqueeze(1)).sum()).item()
    return correct_num, predicted.shape[0]



def eval_audio(gpu_id, cfg, ckpt, is_ddp, save_path=None, eval_method="Ours"):
    os.makedirs(save_path, exist_ok=True)
    print('gpu id:',gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    dataloaders, model = load_model_and_dataloaders(cfg, ckpt, device, is_ddp, eval_method)
    i = 0
    total_correct_list = []
    total_len_list = []
    # import pdb
    # pdb.set_trace()
    # for i in range(100):
    #     print(dataloaders.dataset[i]['video_feat'].shape)
    for batch in tqdm(dataloaders):
        i += 1
        correct_num, sample_len = model_inference(model, batch)
        print("Batch: {}  ACC: {}".format(i, correct_num / sample_len))
        total_correct_list.append(correct_num)
        total_len_list.append(sample_len)
    
    correct_list = np.array(total_correct_list)
    len_list = np.array(total_len_list)

    avg_acc = correct_list.sum() / len_list.sum()
    total_num = len_list.sum()
    return avg_acc, total_num

            

def main():
    eval_method = "Ours" # or SpecVQGAN

    save_path = ""

    # Eval Classifier:
    cfg_path = "./evaluation/align_acc/eval_classifier.yaml"
    ckpt = ""      # put the eval classifier under diff_foley_ckpt

    torch.manual_seed(0)
    local_rank = os.environ.get('LOCAL_RANK')

    if local_rank is not None:
        is_ddp = True
        local_rank = int(local_rank)
        dist.init_process_group("nccl", 'env://', datetime.timedelta(0, 300))
        print(f'WORLDSIZE {dist.get_world_size()} – RANK {dist.get_rank()}')
        if dist.get_rank() == 0:
            print('MASTER:', os.environ['MASTER_ADDR'], ':', os.environ['MASTER_PORT'])
    else:
        is_ddp = False
        local_rank = 0
    

    cfg = OmegaConf.load(cfg_path)
    print("Path:", cfg.data_eval_metric.params.validation.params.eval_dataset_path)
    
    avg_acc, total_num = eval_audio(local_rank, cfg, ckpt, is_ddp, save_path=save_path, eval_method=eval_method)
    print("Metric =====> Avg ACC: {}   Total Num: {}".format(avg_acc, total_num))

    with open(os.path.join(save_path, "results_metric.txt"), "w") as f:
        txt = "AVG ACC: {}   Total Num: {}".format(avg_acc, total_num)
        f.writelines(txt)
    
    print("Path:", cfg.data_eval_metric.params.validation.params.eval_dataset_path)
    


if __name__ == "__main__":
    main()



