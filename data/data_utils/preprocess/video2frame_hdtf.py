import os
import torch
import random
from tqdm import tqdm
import ffmpeg
import pickle
from multiprocessing import Pool
import argparse, os, cv2, traceback, subprocess

import matplotlib
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import sys
from PIL import Image
from glob import glob
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import skimage.io as io
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import torch.nn.functional as F
# import torchvision.utils as vutils
import pdb

import ffmpeg


def get_video_info(input_file_path):

    probe = ffmpeg.probe(input_file_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    frame_rate = float(video_stream["r_frame_rate"].split("/")[0])
    frame_num = int(video_stream["nb_frames"])
    return int(video_stream["height"]), int(video_stream["width"]), frame_rate, frame_num


def multi_preprocess_video(x):
    (iden, output_folder, (height, width, frame_rate, total_frame_num)) = x
        
    input_file_path = os.path.join(vid_dir, iden, "video.mp4")
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(input_file_path)
    print(iden, frame_rate)
    
    count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        file_name = f"{count:0>5}"
        frame_path = os.path.join(output_folder, f"{file_name}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        
        
    return None


def preprocess_video_folder(
    reprocessings, multi_processing, option=None, workers=32
):
    print("Preprocess start !!!")
    if option is None:
        option = {} 

    reprocessings.sort()
    multi_output_frame_path_list = [
        os.path.join(saving_dir, iden) for iden in reprocessings
    ]
    multi_vid_info_list = [
        get_video_info(os.path.join(vid_dir, f"{iden}/video.mp4")) for iden in reprocessings
    ]
 
    def initializer():
        sys.stdout = open(os.devnull, "w")

    if multi_processing:
        """
        for real running
        """
        pool = Pool(workers)
        total = len(reprocessings)
        
        with tqdm(total=total) as pbar:
            pool.imap(multi_preprocess_video, zip(reprocessings, multi_output_frame_path_list, multi_vid_info_list))
            pbar.update()
        
       
        pool.close()
        pool.join()
    return


def read_file(filepath: os.PathLike):
    """
    Reads a file as a space-separated dataframe, where the first column is the index
    """
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        lines = [l.split(":")[0] for l in lines]

    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument("--multi_processing", type=bool, default=True)
    parser.add_argument(
        "--gpu", help="Number of GPUs across which to run in parallel", default=0, type=int
    )
 
    vid_dir = "/media/data1/HDTF_preprocessed/25_fps/"
    saving_dir = "/media/data/HDTF_preprocessed/25_frame/HDTF"
    eval_list = os.listdir(vid_dir)
    
    process_id = []
    for id_ in tqdm(eval_list):
        try:
            vid = f"{vid_dir}/{id_}/video.mp4"
            height, width, frame_rate, frame_num = get_video_info(vid)
            if frame_num != len(glob(os.path.join(saving_dir, id_, '*.jpg'))):
                process_id.append(id_)
        except:
            print(id_)
    
    print(len(process_id))        
    args = parser.parse_args()
    preprocess_video_folder(
        process_id,
        args.multi_processing,
    )
