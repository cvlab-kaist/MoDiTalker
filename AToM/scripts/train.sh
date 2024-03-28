##############
BATCH_SIZE=64
EPOCHS=2000
FEAT=jukebox 
SAVE_INTERVAL=1
DEVICE=6
##############

CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --feature_type $FEAT \
    --save_interval $SAVE_INTERVAL