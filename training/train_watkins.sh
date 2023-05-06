accelerate launch train.py \
--text_encoder_name "laion/clap-htsat-unfused" \
--train_file="data/watkins_train.json" --validation_file="data/watkins_eval.json" --test_file="data/watkins_eval.json" \
--unet_model_config="configs/diffusion_model_config.json" --freeze_text_encoder \
--gradient_accumulation_steps 1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
--learning_rate=3e-5 --num_train_epochs 40 --snr_gamma 5 \
--text_column captions --audio_column location --checkpointing_steps="best" \
--audioldm_model "cvssp/audioldm-s-full-v2" --output_dir "models_watkins_2" --save_every 1 --with_tracking --augment