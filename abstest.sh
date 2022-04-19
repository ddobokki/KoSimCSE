GPU_IDS="0,1,2,3"
PORT_ID=$(expr $RANDOM + 1000)
step_num=250
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node 4 --master_port $PORT_ID unsup_train_with_huggingface.py \
    --output_dir output \
    --model_name_or_path klue/bert-base \
    --train_file data/train/wiki_train.csv \
    --dev_file data/dev/sts_dev.tsv \
    --test_file data/test/sts_test.tsv \
    --per_device_train_batch_size 256 \
    --max_seq_length 32 \
    --pooler_type cls \
    --num_train_epochs 3 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --label_names labels \
    --do_train \
    --do_eval \
    --save_total_limit 3 \
    --fp16 \
    --logging_steps $step_num \
    --save_steps $step_num \
    --eval_steps $step_num \
    --logging_first_step True \
    --include_inputs_for_metrics True \
    --metric_for_best_model spearman \
    "$@"
