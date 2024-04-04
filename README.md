# MoDiTalker

Official PyTorch implementation of **["MoDiTalker: Motion-Disentangled Diffusion
Model for High-Fidelity Talking Head Generation"](https://arxiv.org/abs/2403.19144)**.   
<!-- [Seyeon Kim](https://sihyun.me/)<sup>*1</sup>, 
[Siyoon Jin](https://sites.google.com/site/kihyuksml/)<sup>*1</sup>, 
[Jihye Park](https://subin-kim-cv.github.io/)<sup>1</sup>, 
[Kihong Kim](https://alinlab.kaist.ac.kr/shin.html)<sup>2</sup>,
[Jiyoung Kim]()<sup>1</sup>,
[Jisu Nam]()<sup>1</sup> and
[Seungryong Kim]()<sup>1</sup>. -->
Seyeon Kim<sup>&#8727;1</sup>, 
Siyoon Jin<sup>&#8727;1</sup>, 
Jihye Park<sup>&#8727;1</sup>, 
Kihong Kim<sup>2</sup>,
Jiyoung Kim<sup>1</sup>,
Jisu Nam<sup>1</sup> and
Seungryong Kim<sup>&dagger;1</sup>.
<br>
<sup>&#8727;</sup> Equal contribution ,<sup>&dagger;</sup>Corresponding author
<br>
<sup>1</sup>Korea University, <sup>2</sup>VIVE STUDIO  
[paper](https://arxiv.org/abs/2403.19144) | [project page](https://ku-cvlab.github.io/MoDiTalker/)


## 1. Environment setup

```bash
conda create -n MoDiTalker python=3.8 -y
conda activate MoDiTalker
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install natsort tqdm gdown omegaconf einops lpips pyspng tensorboard imageio av moviepy numba p_tqdm soundfile face_alignemnt
```

## 2. Get ready to train models 

### 2.1. Dataset 
<!-- Currently, we provide experiments for the following two datasets: [LRS3](path to lrs3 or geneface) and [HDTF](https://github.com/MRzzm/HDTF). Each dataset is used for training AToM and MToV, respectively. Please refer the README.md in `/data`. Each dataset should be placed in `/data` with the following structures below; -->
We utilized two datasets for training each stage. 
Please refer and follow the dataset preparation from [here](https://github.com/KU-CVLab/MoDiTalker/data/README.md)


### 2.2. Download auxiliary models
<!-- Download  [this link](https://drive.google.com/file/d/1d08qauPUH0Nu_yN2gcmreLSiOiweD5OE/view?usp=sharing) -->
Get the [`BFM_model_front.mat`](https://drive.google.com/file/d/1d08qauPUH0Nu_yN2gcmreLSiOiweD5OE/view?usp=sharing), [`similarity_Lm3D_all.mat`](https://drive.google.com/file/d/17zp_zuUYAuieCWXerQkbp8SRSU4KJ8Fx/view?usp=sharing) and [`Exp_Pca.bin`](https://drive.google.com/file/d/1SPeJ4jcJT9VS4IdA7opzyGHCYMKuCLRh/view?usp=sharing), and place them to the `MoDiTalker/data/data_utils/deep_3drecon/BFM` directory.
Obtain ['BaselFaceModel.tgz](https://drive.google.com/file/d/1Kogpizrcf2zTm1fX9uUUWZuMQqHM7DOc/view?usp=sharing) and extract a file named `01_MorphableModel.mat` and place it to the `MoDiTalker/data/data_utils/deep_3drecon/BFM` directory.

#### (Optional) 
We had to revise a single code inside the package `accelerate` due to version conflicts. If some conflicts occur during loading data, please revise the code in `accelerate/dataloader.py` following this;

from 
```bash
batch_size = dataloader.batch_size if dataloader.batch_size is not None else dataloader.batch_sampler.batch_size
```
to
```bash
batch_size = dataloader.batch_size if dataloader.batch_size is not None else len(dataloader.batch_sampler[0])
```



## 3. Training

### 3.1. AToM

```bash
cd AToM
bash scripts/train.sh
```
The checkpoints of AToM will be saved in `./runs`

### 3.2. MToV
The checkpoints of AToM will be saved in `./runs`

### Autoencoder

First, execute the following script:
```bash
cd MToV
bash scripts/train/first_stg.sh 
```
Then the script will automatically create the folder in `./log_dir` to save logs and checkpoints.

Second, execute the following script:
```bash
cd MToV
bash scripts/train/first_stg_ldmk.sh 
```
You may change the model configs via modifying `configs/autoencoder`. Moreover, one needs early-stopping to further train the model with the GAN loss (typically 8k-14k iterations with a batch size of 8).

### Diffusion model
```bash
cd MToV
bash scripts/train/second_stg.sh
```


### 4. Getting the Weights
We provide the corresponding checkpoints in the below:
Download and place them in the `./checkpoints/` directory. 

Full checkpoints will be released later, ETA July 2024.

### 5. Inference
#### 5.1. Generating Motions from Audio 
Before producing the motions from audio, there's need to preprocess the audio since we process audio in the type of HuBeRT. To produce hubert feature of audio you want, please follow the script below:

```bash
cd data
python data_utils/preprocess/process_audio.py \
--audio path to audio \
--ref_dir path to directory of reference images 
```

Then the processed audio hubert(npy) will be saved in `data/inference/hubert/{sampling rate}` 

Note that, you need to specify the path to (1) reference images (2) processed hubert and (3) checkpoint in the following bash script. 

```bash
cd AToM
bash scripts/inference.sh
```

The results of AToM will be saved in `AToM/results/frontalized_npy` and this path should be consistent with the `ldmk_path` of the following step.

#### 5.2. Align Motions
Note that, you need to specify the path to (1) reference images and (2) produced landmark. 

```bash 
cd data/data_utils
python motion_align/align_face_recon.py \
--ldmk_path path to directory of generated landmark \
--driv_video_path path to directory of reference images 
```
The final landmarks will be saved in `AToM/results/aligned_npy`.

#### 5.3. Generating Video from aligned Motions
```bash 
cd MToV
bash scripts/inference/sample.sh
```
The final videos will be saved in `MToV/results`.



### Citation
```bibtex
@misc{kim2024moditalker,
      title={MoDiTalker: Motion-Disentangled Diffusion Model for High-Fidelity Talking Head Generation}, 
      author={Seyeon Kim and Siyoon Jin and Jihye Park and Kihong Kim and Jiyoung Kim and Jisu Nam and Seungryong Kim},
      year={2024},
      eprint={2403.19144},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Reference
This code is mainly built upon [EDGE](https://github.com/Stanford-TML/EDGE) and [PVDM](https://github.com/sihyun-yu/PVDM/tree/main).\
We also used the code from following repository: [GeneFace](https://github.com/yerfor/GeneFace).
