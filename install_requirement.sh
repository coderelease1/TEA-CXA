pip3 install accelerate bitsandbytes deepspeed==0.16.4 isort jsonlines loralib optimum peft pynvml>=12.0.0  tensorboard  torchmetrics  transformers_stream_generator wandb

pip3 install "qwen-agent[code_interpreter]"
pip3 install llama_index bs4 pymilvus infinity_client codetiming tensordict==0.6 omegaconf torchdata==0.10.0 hydra-core easydict mcp==1.9.3
pip3 install -e . --no-deps
pip3 install faiss-gpu-cu12   # Optional, needed for end-to-end search model training with rag_server