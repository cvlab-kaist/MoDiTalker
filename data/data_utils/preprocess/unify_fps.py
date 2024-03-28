import os
import cv2
import glob
import time
import torch
import random
import shutil
from tqdm import tqdm
import matplotlib
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse, os, cv2, traceback, subprocess
import pdb
import sys
import subprocess

def change_video_fps(input_path, output_path, fps):
    command = f'ffmpeg -i {input_path} -r {fps} {output_path}'
    subprocess.call(command, shell=True)
    
class Dataset_(Dataset):
    def __init__(self,args):
        self.args=args
        self.device = torch.device('cuda')
        self.total_idx = 0
        self.start_time = time.time()

        self.path_to_mp4   = args.load_video_path
        self.path_to_frame = args.save_video_path
        
        os.makedirs(self.path_to_frame, exist_ok=True)
        
        self.video_path_list= []
        self.frame_dir_path_list= []

        self.total_num_list = []
        self.initList()
    
    
    def initList(self):
        length = 0
        videos = glob.glob(os.path.join(self.path_to_mp4, '*', "*audio.mp4")) 
        videos.sort()
    
            
        for video in videos:
            self.frame_dir_path_list.append(video.replace('videos/','videos_25/'))

        self.video_path_list = videos
        self.frame_dir_path_list = self.frame_dir_path_list
        self.length = len(self.video_path_list)

    def change_video_fps(self, fps, input_path, output_path):
        template = 'ffmpeg -y -i {} -c:v libx264 -r {} {}'
        command = template.format(input_path, fps, output_path)
        subprocess.call(command, shell=True)

    def generate_and_save_frame(self, idx):
        input_path = self.video_path_list[idx]
        output_path = self.frame_dir_path_list[idx]
        os.makedirs(os.path.dirname(os.path.dirname(output_path)), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.change_video_fps(args.fps, input_path, output_path)
        
        return 0 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_time = time.time()
        self.generate_and_save_frame(idx)
    
        return [0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--load_video_path', type=str, default='/media/data1/HDTF/videos', # Source Video Roots
                            help='path of the directory for loading videos')
    parser.add_argument('--save_video_path', type=str, default='/media/data1/HDTF/videos_25', # Saving Roots
                            help='path of the directory for saving frames of videos')

    parser.add_argument('--fps', type=int, default=25,
                            help='fps')
    parser.add_argument('--batch_size', type=int, default=1,
                            help='audio sampling rate')
    parser.add_argument('--num_workers', type=int, default=6,
                            help='audio sampling rate')
    args = parser.parse_args()
    count = 0
    
    dataset = Dataset_(args)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    start_time = time.time()
    for i, video_path in enumerate(tqdm(data_loader)):
        video_path = video_path
        dataset.generate_and_save_frame(video_path)
    print('done')