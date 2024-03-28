import argparse
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch
import pdb
import glob, os, tqdm

@torch.no_grad()
def get_hubert_from_speech(args, speech):
    device = args.device
    print(torch.cuda.is_available())
    print("Loading the Wav2Vec2 Processor...")
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    print("Loading the HuBERT Model...")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    # global hubert_model
    hubert_model = hubert_model.to(device)
    if speech.ndim == 2:
        speech = speech[:, 0]  # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]
    input_values_all = input_values_all.to(device)
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx:end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter :]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it
        hidden_states = hubert_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

def convert_wav_sampling_rate(args):
    save_root = args.save_sample_dir
    source_wav_name = args.audio.split("/")[-1].split(".")[0]
    supported_types = (".wav", ".mp3", ".mp4", ".avi") 
    os.makedirs(os.path.join(save_root, str(args.sampling_rate)), exist_ok = True)
    new_wav_name = os.path.join(save_root, str(args.sampling_rate), f"{source_wav_name}")
    command = f"ffmpeg -i {args.audio} -f wav -ar {args.sampling_rate} {new_wav_name}.wav -y"
    os.system(command)
    
def load_idlist(path):
    with open(path, "r") as f:
        lines = f.readlines()
        id_list = [line.replace("\n", "").replace(".mp4", "").strip() for line in lines]
    return id_list

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="audio sampling match")
    ### for sampling audio
    args.add_argument("--audio", type=str, 
                      default="../inference/audio/LetItGo1.wav", help="path to the audio")
    args.add_argument("--save_sample_dir", type=str, 
                      default="../inference/sampled_audio", help="save path to the directory of sampled_audio")
    args.add_argument("--ref_dir", type=str, 
                      default="../inference/ref/25fps", help="path to the directory of reference images")
    args.add_argument("--ref_id_list", type=str, 
                      default=None, 
                      help="if ref_id_list is None, then the whole id in the ref_dir will be included")
    args.add_argument("--sampling_rate", type=int,
                      default=16000)
    args.add_argument("--device", type=str,
                      default="cuda:5")
    
    ### for extracting hubert
    args.add_argument("--wav2vec_proc", type=str,
                      default="facebook/hubert-large-ls960-ft",
                      help="the pretrained wav2vec2 processor")
    args.add_argument("--hubert_model", type=str,
                      default="facebook/hubert-large-ls960-ft",
                      help="the pretrained hubert model")
    args.add_argument("--save_hubert_dir", type=str, 
                      default="../inference/hubert", help="save path to the directory of converted hubert")
    
    args = args.parse_args()
    
    # load id list
    if args.ref_id_list is None: 
        ref_list = os.listdir(args.ref_dir)
    else :
        ref_list = load_idlist(args.ref_id_list)
    
    # convert sampling rate
    convert_wav_sampling_rate(args)
    
    # extract hubert
    # confirm the sampled audio 
    audioname = args.audio.split("/")[-1].split(".")[0] # LetItGo
    
    sampled_audio = os.path.join(args.save_sample_dir, str(args.sampling_rate), f"{audioname}.wav")
    # AToM/data/sampled_audio/19200/LetItGo.wav
    if not os.path.exists(sampled_audio):
        pass # RunTimeError
    hubert_dir = os.path.join(args.save_hubert_dir, str(args.sampling_rate))
    os.makedirs(hubert_dir, exist_ok=True)
    hubert_name = os.path.join(hubert_dir, f"{audioname}.npy")
    speech_, _ = sf.read(sampled_audio)
    hubert_ = get_hubert_from_speech(args, speech_)
    np.save(hubert_name, hubert_.detach().numpy())
    print("Finished preprocessing audio.\n")