import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1,2,3'

import torch
import torch.nn as nn
import time
import datetime
import argparse
import random
import numpy as np

# from tqdm import tqdm
from torch.utils.data import DataLoader
from video_encoder import CAVP_Inference, TrainDataset_v2, collate_fn_v2, CosineSimilarityLoss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_ckpt_dir', type=str, default='')
    parser.add_argument('--processed_data_dir', type=str, default='')
    parser.add_argument('--cavp_ckpt_path', type=str, default='')
    parser.add_argument('--video_type', type=str, default='', help="okk")
    # vgg+crop_align
  
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-04)  
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(42)
    args = parse_args()

    print("save_ckpt_dir: ", args.save_ckpt_dir)
    os.makedirs(args.save_ckpt_dir, exist_ok=True)

    dataset = TrainDataset_v2(processed_data_dir=args.processed_data_dir, video_type=args.video_type)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_v2, shuffle=True, num_workers=args.num_workers)

    CAVP_model = CAVP_Inference(video_encode='Slowonly_pool', embed_dim=512, ckpt_path=args.cavp_ckpt_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        CAVP_model = nn.DataParallel(CAVP_model)
    CAVP_model.to(device)

    print("lr: ", args.lr)
    optimizer = torch.optim.Adam(CAVP_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    cosine_loss = CosineSimilarityLoss()

    # 检查是否需要恢复训练
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        CAVP_model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        print(f"Resumed from checkpoint at epoch {start_epoch}, global step {global_step}")
    
    loss_2 = 0
    start_time = datetime.datetime.now()
    print(f"training begins at : {start_time:%Y-%m-%d %H:%M:%S}")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        for step, batch in enumerate(dataloader):
        # for step, batch in enumerate(tqdm(dataloader, desc="Batches", unit="batch", leave=False)):
            
            global_step = global_step + 1
        
            crop_videos = batch["cavp_inputs"].to(device)
            crop_feats = CAVP_model(crop_videos)
            
            gt_feat = batch["video_feats"].to(device)
            mse_loss = criterion(crop_feats, gt_feat)
            cos_loss = cosine_loss(crop_feats, gt_feat)
            total_loss = mse_loss + cos_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_2 = loss_2 + total_loss.item()
            if global_step % 2000 == 0:
                print(f"step {global_step}, step_Loss: {loss_2}")
                loss_2 = 0
        
            # if global_step % 500 == 0:
            #     save_path = os.path.join(save_path_dir, f"checkpoint-{global_step}")
            #     torch.save(CAVP_model.state_dict(), save_path)

        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Epoch [{epoch+1}/{args.epochs}], epoch_Loss: {total_loss.item()}, time: {total_time} s")
        save_path = os.path.join(args.save_ckpt_dir, f"checkpoint-{epoch+1}")
        # torch.save(CAVP_model.state_dict(), save_path)
        # torch.save(CAVP_model.module.state_dict(), save_path)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': CAVP_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step
        }, save_path)
        print(f"|| save checkpoint-{epoch+1}")

    end_time = datetime.datetime.now()
    print(f"training ends at : {end_time:%Y-%m-%d %H:%M:%S}")
        


if __name__ == "__main__":
    main()