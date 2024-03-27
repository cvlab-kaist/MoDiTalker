EXP_NAME=main
DATASET=HDTF
FIRST_STAGE_MODEL_DIRECTORY=../checkpoints/autoencoder_rgb.pth
BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=0 python main.py \
--exp first_stage_ldmk \
--log_dir ./runs \
--id encoder_decoder_frz \
--typetype 'ldmk' \
--timesteps 16 \
--pretrain_config configs/autoencoder/base.yaml \
--data ${DATASET} \
--first_model ${FIRST_STAGE_MODEL_DIRECTORY} \
--batch_size ${BATCH_SIZE}
