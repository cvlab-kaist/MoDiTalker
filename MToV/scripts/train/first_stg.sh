EXP_NAME=main
DATASET=HDTF
BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=6 python main.py \
--exp first_stage \
--id main \
--log_dir ./runs \
--timesteps 4 \
--pretrain_config configs/autoencoder/base.yaml \
--data ${DATASET} \
--batch_size ${BATCH_SIZE}
