# Dataset download & Preprocessing
## LRS3
Follow [here](https://github.com/yerfor/GeneFace/blob/main/docs/process_data/process_lrs3.md) to preprocess LRS3. 

## HDTF

### Download 
Follow [here](https://github.com/universome/HDTF) to download HDTF and crop videos into 256x256 resolution.

## Structure
 After following the steps, the structure should be like this:
```
data
|-- train
    |-- lrs3
        |-- sizes_train.npy
        |-- sizes_val.npy
        |-- spk_id2spk_idx.npy
        |-- train.data
        |-- val.data

    
    |-- HDTF
        |-- frames
            |-- id1
                |-- 00000.jpg
                |-- 00001.jpg
                |-- ...
            |-- id2
                |-- 00000.jpg
                |-- 00001.jpg
                |-- ...
            |-- ...
        |-- keypoints
            |-- face-centric
                |-- posed
                    |-- id1
                        |-- 00000.npy
                        |-- 00001.npy
                        |-- ...
                    |-- id2
                        |-- 00000.npy
                        |-- 00001.npy
                        |-- ...
                    |-- ...
                |-- unposed
                    |-- id1
                        |-- 00000.npy
                        |-- 00001.npy
                        |-- ...
                    |-- id2
                        |-- 00000.npy
                        |-- 00001.npy
                        |-- ...
                    |-- ...
            |-- non-face-centric
                |-- posed
                    |-- id1
                        |-- 00000.npy
                        |-- 00001.npy
                        |-- ...
                    |-- id2
                        |-- 00000.npy
                        |-- 00001.npy
                        |-- ...
                    |-- ...

```

### Video 2 Frames
Before you convert videos into frames, check all videos are at 25fps. 
If not, they must be adjusted to: `data/data_utils/preprocess/unify_fps.py`
Once preprocesing is complete and the videos are unitied at 25fps, you can convert them into frames using the following code.: `data/data_utils/preprocess/video2frame_hdtf.py`

### Motion Extraction from frames, used in training MToV


### Environment
<!-- ```bash
conda create -n geneface python=3.8 -y
conda activate geneface
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install natsort tqdm gdown omegaconf einops lpips pyspng tensorboard imageio av moviepy numba p_tqdm soundfile face_alignemnt
``` -->
```bash
conda create -n preprocess python=3.9.16 -y
conda activate preprocess
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch 
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d==0.7.4 -c pytorch3d -y
conda install ffmpeg 
python -m pip install face_alignment einops trimesh natsort
```

```bash 
conda activate preprocess
cd data/data_utils
python preprocess/process_video_3dmm_rollback_hdtf_batchify.py
```
After processing the code above, you will obtain the keypoints in several types in `HDTF/keypoints`. `face-centric` and `non-face-centric` indicate whether the keypoints are aligned in the center or not. `unposed` and `posed` specify whether the pose of the landmarks is frontalized or not. 

