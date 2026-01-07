#!/bin/bash
set -x
MODEL=Qwen3-4B-Base
DATA_DIR=/data0/jzzhang/datasets/PR_box
ResultDir="/data0/jzzhang/RLPR/results"
N_GPUS_PER_NODE=8
MAX_TOKENS=32768
MAX_ROLLOUT_TOKENS=32768
EXP_NAME=Qwen3-4B_rlhr_cot_answer_topk_clip01_box
layer=29

USE_WANDB=${USE_WANDB:-"false"}
export ACCELERATE_LOG_LEVEL=info
# export NCCL_P2P_DISABLE=1
WANDB_PRJ_NAME=rlpr
unset OPENAI_API_KEY
unset OPENAI_API_BASE
export USED_MODEL=${USED_MODEL:-"no_api"}

TRAIN_FILES=$DATA_DIR/rlpr_train.parquet
VAL_DIR=$DATA_DIR
VAL_FILES=[${VAL_DIR}'/Math-500_Avg2.parquet',${VAL_DIR}'/gpqa_diamond_Avg4.parquet',${VAL_DIR}'/AIME2024_Avg16.parquet',${VAL_DIR}'/WebInstruct-verified-val_Avg2.parquet',${VAL_DIR}'/Minerva_Avg4.parquet',${VAL_DIR}'/TheoremQA_Avg2.parquet',${VAL_DIR}'/MMLUProALL.parquet',${VAL_DIR}'/SuperGPQA.parquet',${VAL_DIR}'/AMC.parquet',${VAL_DIR}'/AIME25.parquet']

# Logging and Checkpointing
export LOGS_PATH=${ResultDir}/data/logs
export TENSORBOARD_DIR=${ResultDir}/tensorboard/${EXP_NAME}
mkdir -p "${TENSORBOARD_DIR}"
VAL_SAVE_RESULTS_DIR=${ResultDir}/data/logs/test_generations_${EXP_NAME}
mkdir -p "${VAL_SAVE_RESULTS_DIR}"
LOCAL_DIR=${ResultDir}/data/checkpoints/${EXP_NAME}
mkdir -p "${LOCAL_DIR}"

# --- Conditional WandB Setup ---
TRAINER_LOGGER_CONFIG="['console']" # Default logger
declare -a WANDB_PARAMETERS # Array to hold WandB specific parameters

if [ "$USE_WANDB" = "true" ]; then
    echo "WandB logging ENABLED. Make sure you have logged in."
    export WANDB_MODE=offline
    export WANDB_DIR_PATH=${ResultDir}/wandb/${EXP_NAME}
    mkdir -p "${WANDB_DIR_PATH}"

    export WANDB_DIR=${WANDB_DIR_PATH}

    TRAINER_LOGGER_CONFIG="['console','wandb','tensorboard']"
    WANDB_PARAMETERS=(
        "trainer.project_name=$WANDB_PRJ_NAME"
        "trainer.val_generations_to_log_to_wandb=10"
        "+trainer.train_generations_to_log_to_wandb=1"
        "+trainer.train_generations_to_log_to_wandb_2=50"
        "+wandb_dir=${WANDB_DIR}" # Use the exported WANDB_DIR which is ./wandb
    )
else
    echo "WandB logging DISABLED."
    unset WANDB_DIR
fi

export HYDRA_FULL_ERROR=1
nnodes=${VERL_N_TRAIN_NODE:-1}
KL_COEF=0

# Main Training Command and Configuration
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_ROLLOUT_TOKENS \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.merge_golden_method=cot_answer \
    +actor_rollout_ref.actor.clip_ratio_low=0.2 \
    +actor_rollout_ref.actor.clip_ratio_high=0.27 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=${TRAINER_LOGGER_CONFIG} \
    trainer.experiment_name=$EXP_NAME \
    "${WANDB_PARAMETERS[@]}" \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=$nnodes \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    +trainer.test_decoding_strategy=sample \
    +actor_rollout_ref.rollout.val_temperature=0.6 \
    +actor_rollout_ref.rollout.val_top_p=0.7 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=300 \
    +trainer.val_save_results_dir=${VAL_SAVE_RESULTS_DIR} \
    trainer.default_local_dir=${LOCAL_DIR} \
    trainer.resume_mode=disable \
    reward_model.reward_manager=hidden \
    +reward_model.repetition_penalty=False \
    +reward_model.val_reward_manager=naive \
    +reward_model.format_mode=boxed \
    actor_rollout_ref.actor.is_get_hidden=True \
    +actor_rollout_ref.actor.reward_hidden_type=subspace_energy_overlap_topk \
    actor_rollout_ref.actor.layer_list=[$layer] \
    +reward_model.reward_manager_shaping_function_name=clip_01 \
    +reward_model.format_coefficient=0 \
    "$@"