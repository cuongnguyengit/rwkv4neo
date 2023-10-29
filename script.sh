

# train from scratch
!python train.py --load_model "" \
    --wandb "" --proj_dir "/content/drive/MyDrive/llm/checkpoint/rwkv4_20231028/" \
    --data_file "/content/binidx/mydataset_text_document" --data_type "binidx" --vocab_size 20000 \
    --ctx_len 128 --epoch_steps 10000 --epoch_count 11 --epoch_begin 9 --epoch_save 1 \
    --micro_bsz 128 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 \
    --lr_init 6e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0


# full finetune from world 0.1B-L12-768-ctx4096 model
!python train.py --load_model "/content/drive/MyDrive/llm/checkpoint/RWKV-4-World-0.1B-v1-20230520-ctx4096.pth" \
    --wandb "" --proj_dir "/content/drive/MyDrive/llm/checkpoint/rwkv4_0.1B_full_finetune_8B_20231029/" \
    --data_file "/content/binidx/multilingual_text_document" --data_type "binidx" --vocab_size 65536 \
    --ctx_len 4096 --epoch_steps 10000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 1 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 \
    --lr_init 5e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0


# full finetune from world 0.4B-L24-1024-ctx4096 model
!python train.py --load_model "/content/drive/MyDrive/llm/checkpoint/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth" \
    --wandb "" --proj_dir "/content/drive/MyDrive/llm/checkpoint/rwkv4_0.4B_full_finetune_8B_20231029/" \
    --data_file "/content/binidx/multilingual_text_document" --data_type "binidx" --vocab_size 65536 \
    --ctx_len 4096 --epoch_steps 10000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 1 --n_layer 24 --n_embd 1024 --pre_ffn 0 --head_qk 0 \
    --lr_init 5e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0


# full finetune from world 1.5B-L24-2048-ctx4096 model
!python train.py --load_model "/content/drive/MyDrive/llm/checkpoint/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth" \
    --wandb "" --proj_dir "/content/drive/MyDrive/llm/checkpoint/rwkv4_1.5B_full_finetune_8B_20231029/" \
    --data_file "/content/binidx/multilingual_text_document" --data_type "binidx" --vocab_size 65536 \
    --ctx_len 4096 --epoch_steps 10000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 1 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    --lr_init 5e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0


# full finetune from world 3B-L32-2560-ctx4096 model
!python train.py --load_model "/content/drive/MyDrive/llm/checkpoint/RWKV-4-World-3B-v1-20230619-ctx4096.pth" \
    --wandb "" --proj_dir "/content/drive/MyDrive/llm/checkpoint/rwkv4_3B_full_finetune_8B_20231029/" \
    --data_file "/content/binidx/multilingual_text_document" --data_type "binidx" --vocab_size 65536 \
    --ctx_len 4096 --epoch_steps 10000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 1 --n_layer 32 --n_embd 2560 --pre_ffn 0 --head_qk 0 \
    --lr_init 5e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0


# lora finetune from world 1.5B-L24-2048-ctx4096 model
!python train_lora.py --load_model "/content/drive/MyDrive/llm/checkpoint/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth" \
    --wandb "" --proj_dir "/content/drive/MyDrive/llm/checkpoint/rwkv4_1.5B_full_finetune_8B_20231029/" \
    --data_file "/content/binidx/multilingual_text_document" --data_type "binidx" --vocab_size 65536 \
    --ctx_len 4096 --epoch_steps 10000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 1 --n_layer 24 --n_embd 2048 --pre_ffn 0 --head_qk 0 \
    --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \
    --lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.01 \
    --lora_parts=att,ffn,time,ln


# lora finetune from world 3B-L32-2560-ctx4096 model
!python train_lora.py --load_model "/content/drive/MyDrive/llm/checkpoint/RWKV-4-World-3B-v1-20230619-ctx4096.pth" \
    --wandb "" --proj_dir "/content/drive/MyDrive/llm/checkpoint/rwkv4_3B_full_finetune_8B_20231029/" \
    --data_file "/content/binidx/multilingual_text_document" --data_type "binidx" --vocab_size 65536 \
    --ctx_len 4096 --epoch_steps 10000 --epoch_count 1 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 1 --n_layer 32 --n_embd 2560 --pre_ffn 0 --head_qk 0 \
    --lr_init 1e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \
    --lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.01 \
    --lora_parts=att,ffn,time,ln