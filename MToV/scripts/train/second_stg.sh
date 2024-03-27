EXP_NAME=main
DATASET=HDTF
FIRST_STAGE_MODEL_DIRECTORY=../checkpoints/autoencoder_rgb.pth
FIRST_STAGE_MODEL_LDMK_DIRECTORY=../checkpoints/autoencoder_motion.pth
BATCH_SIZE=10

CUDA_VISIBLE_DEVICES=6 python main.py \
--exp ddpm \
--id ${EXP_NAME} \
--log_dir ./runs \
--data ${DATASET} \
--first_model ${FIRST_STAGE_MODEL_DIRECTORY} \
--first_model_ldmk ${FIRST_STAGE_MODEL_LDMK_DIRECTORY} \
--pretrain_config configs/autoencoder/base.yaml \
--diffusion_config configs/latent-diffusion/base.yaml \
--batch_size ${BATCH_SIZE}

