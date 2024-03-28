###############
DATA_ROOT=../data/inference/ref/25fps
HUBERT=../data/inference/hubert/16000/LetItGo1.npy
SAVE_DIR=results/frontalized1
CHECKPOINT=../checkpoints/atom.pt
DEVICE=6
###############

CUDA_VISIBLE_DEVICES=$DEVICE python inference.py \
    --data_root $DATA_ROOT \
    --hubert_path $HUBERT \
    --save_dir $SAVE_DIR \
    --checkpoint $CHECKPOINT