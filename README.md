# Which Tool Response Should I Trust? Tool-Expertise-Aware Chest X-ray Agent with Multimodal Agentic Learning
This is an anonymous repository for our method TEA-CXA based on PyTorch. We use one node of 8 H800 GPUS.

Since we are using a SLURM system, we provide the instructions that align with SLURM system. Nonetheless, the commands used can be extended to non-SLURM systems as well.

## Tool weights downloading
Define the `CACHE_DIR` variable (i.e., your path to folder storing tool model weights) in `server/server_medgemma.py` and `server/server_lingshu.py`. Then run the following tool deployment commands that will automatically download the tool model weights. 
```
CUDA_VISIBLE_DEVICES=0 python server/server_lingshu.py --port 5006
CUDA_VISIBLE_DEVICES=0 python server/server_medgemma.py --port 5007
```
After the downloading and tool deployment are completed, modify the test image paths (the `image_path_list` variables) and the name of the node where the two tools are running (the default value of `SERVERS_HOST`) in `server/test_servers.py`, and then run `python server/test_servers.py` to test if the tool deployment is successful. Then you can terminate the scripts .

## Training


Change the `conda.sh` path, `MODEL_PATH`, and `RESULT_DIR` in `train.slurm`. Then run `train.slurm` as follows.
```
sbatch train.slurm
```