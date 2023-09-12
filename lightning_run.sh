# Use distributed data parallel
CUDA_VISIBLE_DEVICES=1,4,6,7 python lightning_pretrain.py \
    --model_name_or_path allenai/longformer-base-4096 \
    --train_file pretrain_data/train.json \
    --dev_file pretrain_data/dev.json \
    --item_attr_file pretrain_data/meta_data.json \
    --output_dir result/recformer_pretraining \
    --num_train_epochs 32 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8  \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --device 4 \
    --fp16 \
    --fix_word_embedding