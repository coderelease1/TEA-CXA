import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import traceback
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from threading import Lock

class MedGemmaServerEngine:
    DEFAULT_MODEL_NAME = "google/medgemma-4b-it"
    DEFAULT_DEVICE = "cuda:0"
    DEFAULT_DTYPE = torch.bfloat16
    CACHE_DIR = '/your/path/to/model-weights'


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


        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",#self.device
            cache_dir=self.cache_dir,
        )
        #self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)


        self.model.eval()

        self.gen_lock = Lock()

        print(f"Loaded model: {self.model_name}")
        print(f"Device: {self.device}, DType: {self.dtype}")

    def _build_messages(self, img: Image.Image, prompt: str):

        #user_content = [{"type": "text", "text": prompt}]
        #user_content.append({"type": "image", "image": img})
        user_content = [{"type": "image", "image": img}]
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}],
            },
            {"role": "user", "content": user_content},
        ]
        return messages

    @torch.inference_mode()
    def _generate_response(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        pil_images = []
        pil_images.append(Image.open(image_path).convert("RGB"))

        messages = self._build_messages(pil_images[0], prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)#, dtype=self.dtype

        #inputs = {k: v.to(self.model.device, dtype=self.dtype) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with self.gen_lock:
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )[0]

        input_len = inputs["input_ids"].shape[-1]
        generation = generation[input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def run_inference(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        try:
            ###
            prompt = prompt + '\nAnswer concisely.'
            if not isinstance(image_path, str):
                raise ValueError("image_paths must be a string.")
            
            """ for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}") """

            response = self._generate_response(image_path, prompt, max_new_tokens)
            return {"response": response}
        except Exception as e:
            trace_info = traceback.format_exc()
            print(f"Error during MedGemma VQA inference for '{image_path}' with prompt '{prompt}': {str(e)}\n{trace_info}")
            return {"error": f"{str(e)}"}


from fastapi import FastAPI

app = FastAPI()

class VQARequest(BaseModel):
    image_path: str
    prompt: str
    max_new_tokens: int = 150

medgemma_engine = MedGemmaServerEngine()

@app.post("/medgemma")
def vqa(request: VQARequest):
    result = medgemma_engine.run_inference(
        image_path=request.image_path,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma FastAPI Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5007, 
        help="Port to run the server on (default: 5007)"
    )

    args = parser.parse_args()

    # Launch the server on http://0.0.0.0:5007
    print(f"Starting MedGemma Server on 0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)