NUM_FRAMES=144 # NUM_FRAMES=304 = 10초  16의 배수
FIRST_STAGE_MODEL_DIRECTORY=../checkpoints/autoencoder_rgb.pth
FIRST_STAGE_MODEL_LDMK_DIRECTORY=../checkpoints/autoencoder_motion.pth
SECOND_STAGE_MODEL_DIRECTORY=../checkpoints/diffusion_model.pth
EVAL_NAME=noisy_0.25


CUDA_VISIBLE_DEVICES=6 python sample.py \
--including_ldmk_video \
--ratio_ 0.25 \
--fps 30 \
--seconds 5 \
--x_noisy_start \
--num_frames ${NUM_FRAMES} \
--batch_size 1 \
--first_model ${FIRST_STAGE_MODEL_DIRECTORY} \
--first_model_ldmk ${FIRST_STAGE_MODEL_LDMK_DIRECTORY} \
--second_model ${SECOND_STAGE_MODEL_DIRECTORY} \
--eval_folder eval_results/ae_tuned/${EVAL_NAME} 
# --ldmk_owner_list WDA_BarbaraLee1_000 WDA_StenyHoyer_000 \
# --crossID WDA_BarackObama_001 \