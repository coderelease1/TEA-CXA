<!-- # Which Tool Response Should I Trust? Tool-Expertise-Aware Chest X-ray Agent with Multimodal Agentic Learning -->
# TEA-CXA
This is an anonymous repository for our method TEA-CXA based on PyTorch. In our paper, we use one node of 8 H800 GPUs.<br>
Our codebase is built on [RL-Factory](https://github.com/Simple-Efficient/RL-Factory), with the following main improvements:<br>
1. Reinforcement learning with tool calling, featuring robust and verifiable multimodal support, enabling both the policy model and tools to accept multimodal inputs.
2. A machanism allowing for image selection when invoking tools on multi-image queries.
3. Capability to issue multiple tool calls in a single turn.
4. Deployment of multiple instances of a tool server to enable parallel tool inference.

Since we are using a SLURM system, we provide the instructions that align with SLURM system. Nonetheless, the commands used can be extended to non-SLURM systems as well.

## Datasets
Download the three subsets of [CheXbench](https://arxiv.org/pdf/2401.12208) used in our paper:<br>
[Rad-Restruct](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz)<br>
[SLAKE](https://www.med-vqa.com/slake/)<br>
[OpenI](http://openi.nlm.nih.gov/)<br>
We have prepared the metadata files of one fold in our experiments under `./data`. You need to modify the image path prefixes.

## Environment
Key dependencies:
```
Cuda: 12.4
Python: 3.10
vllm: 0.8.5
```
To install requirements, we provide a conda environment.
```
conda env create -f environment.yml
conda activate tea
pip install -e . --no-deps
```
If you prefer to use pip to install the requirements, you can also refer to the environment installment in [RL-Factory](https://github.com/Simple-Efficient/RL-Factory).

## Tool weights downloading
Define the `CACHE_DIR` variable (i.e., your path to folder storing tool model weights) in `server/server_medgemma.py` and `server/server_lingshu.py`. Then run the following tool deployment commands that will automatically download the tool model weights. 
```
CUDA_VISIBLE_DEVICES=0 python server/server_lingshu.py --port 5006
CUDA_VISIBLE_DEVICES=0 python server/server_medgemma.py --port 5007
```
After the downloading and tool deployment are completed, modify the test image paths (the `image_path_list` variables) and the name of the node where the two tools are running (the default value of `SERVERS_HOST`) in `server/test_servers.py`, and then run `python server/test_servers.py` to test if the tool deployment is successful. Then you can terminate the scripts .

## Training

- Set the `conda.sh` path, `MODEL_PATH`, and `RESULT_DIR` in `joint.slurm`.
- Replace the path prefix to `servers_node.txt` in `./envs/tools/classification.py` (line 23) with your working directory.
- Run `joint.slurm` as follows.
```
sbatch joint.slurm
```

## Results
Overall accuracy: 0.7670.<br>
(Note that the overall accuracies of the two individual tools for this fold are:<br> MedGemma: 0.6650; Lingshu: 0.7379.)

## Tool server host on non-SLURM systems
In our experiments we use the node name for passing tool-calling requests to tool servers. If you are using a non-SLURM system, you need to change the host name of tool servers, i.e., the `SERVERS_HOST` variable in `./envs/tools/classification.py`.

## Training on your own tasks and datasets
A guide will be added after the paper is accepted.
