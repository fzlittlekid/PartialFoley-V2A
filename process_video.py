import os
import cv2
import random
from tqdm import tqdm
import torch.multiprocessing as mp

from evaluation.demo_util import reencode_video_with_diff_fps


file_path = ""  # 数据集列表txt路径
split = 'test'
operation_type = 'crop'     # crop or move
save_dir = ""   # 保存路径
dataset_dir = ''    # 数据集路径
tmp_dir = ''    


if operation_type == 'crop':
    folder_name = 'crop_video_4fps' if split == 'train' else 'crop_video'
elif operation_type == 'move':
    folder_name = 'move_video_4fps' if split == 'train' else 'move_video'
save_video_dir = os.path.join(save_dir, split, folder_name)
os.makedirs(save_video_dir, exist_ok=True)



def crop_video(input_video_path, output_video_path):

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 计算裁剪区域的大小
    x_ratio, y_ratio = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
    crop_width = int(width * x_ratio)
    crop_height = int(height * y_ratio)
    # 随机选择裁剪区域的左上角坐标
    x = random.randint(0, int(width - crop_width))
    y = random.randint(0, int(height - crop_height))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_width, crop_height))

    # 逐帧读取视频并裁剪
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[y:y+crop_height, x:x+crop_width]
        out.write(cropped_frame)
        frame_count += 1
        # if frame_count >= fps * 8:  # 8s
        #     break

    # 释放资源
    cap.release()
    out.release()


def move_video(input_video_path, output_video_path):
    
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if abs(total_frame) < 1e-9:
        # return
        raise Exception("the total_frame is zero, please check the video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置输出视频的尺寸大小
    x_ratio, y_ratio = random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)
    output_width = int(width * x_ratio)
    output_height = int(height * y_ratio)

    # 创建输出视频的文件对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    start = random.choice([True, False])
    if width > height:
        x = 0 if start else width - output_width
        y = int((height - output_height) // 2)
        dist = int((width - output_width) // total_frame)
    else:
        x = int((width - output_width) // 2)
        y = 0 if start else height - output_height
        dist = int((height - output_height) // total_frame)

    # 循环处理每一帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = frame[y:y+output_height, x:x+output_width]
        out.write(roi)
        
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

        frame_count += 1
        if frame_count >= fps * 8:  # 8s
            break

    cap.release()
    out.release()


def crop_file(file_name):
    video_path = os.path.join(dataset_dir, file_name + '.mp4')
    output_path = os.path.join(save_video_dir, file_name + '_crop0.mp4')
    if split == 'train':
        video_path_low_fps = reencode_video_with_diff_fps(video_path, tmp_dir, 4, 0, 8)
        crop_video(video_path_low_fps, output_path)
    else:
        crop_video(video_path, output_path)


def move_file(file_name):
    video_path = os.path.join(dataset_dir, file_name + '.mp4')
    output_path = os.path.join(save_video_dir, file_name + '_move0.mp4')
    if split == 'train':
        video_path_low_fps = reencode_video_with_diff_fps(video_path, tmp_dir, 4, 0, 8)
        move_video(video_path_low_fps, output_path)
    else:
        move_video(video_path, output_path)




def main():

    print("preprocess file_list: ", file_path)
    with open(file_path, "r") as f:
        video_list = f.readlines()
        video_list = list(map(lambda x: x.strip(), video_list))

    # crop or move
    print("process Split-{} with operation_type-{}".format(split, operation_type))

    if operation_type == 'crop':
        with mp.Pool(processes=20) as pool:
            results = list(tqdm(pool.imap(crop_file, video_list), total=len(video_list)))
    elif operation_type == 'move':
        with mp.Pool(processes=20) as pool:  
            results = list(tqdm(pool.imap(move_file, video_list), total=len(video_list)))
    else:
        raise Exception("operation_type: crop or move")
        

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()

