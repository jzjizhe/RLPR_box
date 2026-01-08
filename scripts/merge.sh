target_dir=${LOCAL_DIR}/global_step_300_merge
mkdir -p $target_dir
python model_merger.py \
    --local_dir ${LOCAL_DIR}/global_step_300/actor \
    --target_dir  $target_dir
