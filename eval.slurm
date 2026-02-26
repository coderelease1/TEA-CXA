#!/bin/bash

#SBATCH --job-name=eval # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=8             # Number of tasks (GPUs) per node
#SBATCH --gres=gpu:8                    # GPUs per node
#SBATCH --cpus-per-task=8               # CPU cores per GPU
#SBATCH --time=3-00:00:00                 # Maximum runtime
#SBATCH --output=log/train_%j.out           # Standard output
#SBATCH --error=log/train_%j.err            # Standard error
#SBATCH --partition=med # Partition to use
##SBATCH --mail-user= #Update your email address
##SBATCH --mail-type=begin
##SBATCH --mail-type=end

# Load required modules
module load cuda/12.4 slurm

source /your/path/to/conda.sh
#e.g., .../miniconda3/etc/profile.d/conda.sh
conda activate agent_tool

echo $SLURMD_NODENAME > servers_node.txt


CUDA_VISIBLE_DEVICES=0,1 python server/server_lingshu.py --port 5006&
CUDA_VISIBLE_DEVICES=2,3 python server/server_lingshu.py --port 5012&

CUDA_VISIBLE_DEVICES=4 python server/server_medgemma.py --port 5007&
CUDA_VISIBLE_DEVICES=5 python server/server_medgemma.py --port 5009&
CUDA_VISIBLE_DEVICES=6 python server/server_medgemma.py --port 5010&
CUDA_VISIBLE_DEVICES=7 python server/server_medgemma.py --port 5011&

SERVERS_NODE_FILE="servers_node.txt"
if [ ! -f "$SERVERS_NODE_FILE" ]; then
echo "Cannot find $SERVERS_NODE_FILE with servers node hostname."
exit 1
fi
SERVERS_HOST=$(cat $SERVERS_NODE_FILE | tr -d '[:space:]')
echo "Using servers at $SERVERS_HOST"
export SERVERS_HOST="$SERVERS_HOST"

sleep 350

python server/test_servers.py


set -e -x

export HYDRA_FULL_ERROR=1


export DATE=$(date +"%Y-%m-%d-%H:%M:%S")
export RESULT_DIR=/your/path/to/results


export MODEL_PATH=/your/path/to/Qwen2.5-VL-7B-Instruct
export REWARD_MODEL_PATH=Qwen/QwQ-32B
export TEST_DATA='chexbench/splits/fold1_test.jsonl'


python3 -m verl.trainer.main_evaluate --config-name=rl_factory_ppo_trainer \
    data.train_files=chexbench/splits/fold1_test.jsonl\
    data.val_files=$TEST_DATA\
    data.train_batch_size=256\
    data.val_batch_size=2048\
    data.max_prompt_length=5376\
    data.max_response_length=10240\
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.ppo_mini_batch_size=256\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.max_num_batched_tokens=32768\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8\
    actor_rollout_ref.rollout.max_turns=6\
    actor_rollout_ref.rollout.val_kwargs.temperature=0\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1\
    actor_rollout_ref.rollout.val_kwargs.top_p=1\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32\
    actor_rollout_ref.env.name=vision\
    actor_rollout_ref.env.mmtool=True\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.tool_manager=qwen2_5_vl\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.load_custom_chat_template=envs/configs/chat_template.jinja\
    actor_rollout_ref.env.config_path=envs/configs/mcp_vision_tools.pydata\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=2\
    reward_rollout.rollout.gpu_memory_utilization=0.75\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
    reward_rollout.rollout.free_cache_engine=False\
    reward_rollout.rollout.response_length=2048\
    reward_model.reward_manager=parallel\
    trainer.logger=['console','tensorboard']\
    trainer.project_name='eval'\
    trainer.experiment_name="Visual_7B_${DATE}"\
    trainer.n_gpus_per_node=8\
    trainer.nnodes=1\
    trainer.val_only=True\
    trainer.default_local_dir=$RESULT_DIR\
    trainer.default_hdfs_dir=null $@ 2>&1 | tee grpo.log


