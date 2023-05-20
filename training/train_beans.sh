accelerate launch train.py \
--text_encoder_name "laion/clap-htsat-unfused" \
--train_file="data/beans_train_16k.json" --validation_file="data/watkins_eval_processed.json" --test_file="data/beans_classification_eval.json" \
--unet_model_config="configs/diffusion_model_config.json" --freeze_text_encoder \
--gradient_accumulation_steps 1 --per_device_train_batch_size=32 --per_device_eval_batch_size=32 \
--learning_rate=3e-5 --num_train_epochs 200 --snr_gamma 5 \
--text_column captions --audio_column location --checkpointing_steps="best" --load_from_audioldm \
--audioldm_model "cvssp/audioldm-s-full-v2" --output_dir "models_beans_16k" --save_every 1 --with_tracking --augment