GPU_IDS="0,1,2,3"

CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node 4 unsup_train_with_huggingface.py \
    --output_dir output \
    --model_name_or_path klue/bert-base \
    --train_file data/train/wiki_train.csv \
    --dev_file data/dev/sts_dev.tsv \
    --test_file data/test/sts_test.tsv \
    --per_device_train_batch_size 256 \
    --max_seq_length 32 \
    --pooler_type cls \
    --num_train_epochs 1 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --label_names labels \
    --do_train \
    --do_eval \
    --save_total_limit 3 \
    --fp16 \
    --logging_steps 250 \
    --save_steps 250 \
    --eval_steps 250 \
    --logging_first_step True \
    #--metric_for_best_model spearman \
    "$@"
