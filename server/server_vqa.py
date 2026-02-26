import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import traceback


class XRayVQAServerEngine:
    DEFAULT_MODEL_NAME = "StanfordAIMI/CheXagent-2-3b"
    DEFAULT_DEVICE = "cuda:1"
    DEFAULT_DTYPE = torch.bfloat16
    CACHE_DIR = '/your/path/model-weights'

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = self.DEFAULT_MODEL_NAME
        self.device = self.DEFAULT_DEVICE
        self.dtype = self.DEFAULT_DTYPE
        self.cache_dir = self.CACHE_DIR

        # Dangerous code, but works for now
        import transformers
        original_transformers_version = transformers.__version__
        transformers.__version__ = "4.40.0"

        # 加载模型与分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        transformers.__version__ = original_transformers_version

        print(self.model_name, self.model.config._name_or_path, self.model.__class__)
        print(self.tokenizer.init_kwargs.get("cache_dir"))
    def _generate_response(self, image_paths: List[str], prompt: str, max_new_tokens: int) -> str:

        query = self.tokenizer.from_list_format(
            [*[{"image": path} for path in image_paths], {"text": prompt}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            #{"from": "system", "value": "You are a helpful medical assistant." + "Your task is to answer multiple-choice questions about medical images. Please select the single most correct answer." + "\nGive the final choice as a single letter wrapped in <answer></answer>. Example: <answer>A</answer>."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(device=self.device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )[0]
            response = self.tokenizer.decode(output[input_ids.size(1): -1])
            return response
        

    def run_inference(self, image_paths: List[str], prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        try:
            # Verify image paths
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}")

            response = self._generate_response(image_paths, prompt, max_new_tokens)
            return {
                "response": response,
            }
        except Exception as e:
            trace_info = traceback.format_exc()
            print(f"Error during X-ray VQA inference: {str(e)}\n{trace_info}")
            return {
                "error": f"{str(e)}",
            }


#from fastapi import Body

app = FastAPI()

class VQARequest(BaseModel):
    image_paths: List[str]
    prompt: str
    max_new_tokens: int = 512

vqa_engine = XRayVQAServerEngine()

@app.post("/vqa")
def vqa(request: VQARequest):
    result = vqa_engine.run_inference(
        image_paths=request.image_paths,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )
    return result


if __name__ == "__main__":
    # Launch the server on http://0.0.0.0:5004
    print("Starting X-Ray VQA Server on 0.0.0.0:5004")
    uvicorn.run(app, host="0.0.0.0", port=5004)