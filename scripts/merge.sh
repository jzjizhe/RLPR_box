LOCAL_DIR=/data0/jzzhang/RLPR_box/results/Qwen3-4B-Base/data/checkpoints/Qwen3-4B-Base_rlhr_cot_answer_white_norm_layer29_clip01_box
target_dir=${LOCAL_DIR}/global_step_300_merge
mkdir -p $target_dir
python model_merger.py \
    --local_dir ${LOCAL_DIR}/global_step_300/actor \
    --target_dir  $target_dir
