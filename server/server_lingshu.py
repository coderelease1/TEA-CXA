import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import traceback
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from threading import Lock


class LingshuServerEngine:
    DEFAULT_MODEL_NAME = "lingshu-medical-mllm/Lingshu-7B"
    DEFAULT_DEVICE = "cuda:0"
    DEFAULT_DTYPE = torch.bfloat16
    CACHE_DIR = "/your/path/to/model-weights"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        use_flash_attn: bool = True,
    ):
        self.model_name = self.DEFAULT_MODEL_NAME
        self.device = self.DEFAULT_DEVICE
        self.dtype = self.DEFAULT_DTYPE
        self.cache_dir = self.CACHE_DIR

        attn_impl = "flash_attention_2" if use_flash_attn else "eager"

        # load Lingshu-7B（Qwen2.5-VL arch）
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            attn_implementation=attn_impl,
            device_map="auto",
            cache_dir=self.cache_dir,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        self.model.eval()
        self.gen_lock = Lock()

        print(f"Loaded model: {self.model_name}")
        print(f"Device map: auto, DType: {self.dtype}, Attn impl: {attn_impl}")

    def _build_messages(self, img: Image.Image, prompt: str):
        user_content = [{"type": "image", "image": img}]
        user_content.append({"type": "text", "text": prompt})

        """ {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}],
            }, """
        messages = [
            
            {"role": "user", "content": user_content},
        ]
        return messages

    @torch.inference_mode()
    def _generate_response(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        # 打开图像（RGB）
        pil_img = Image.open(image_path).convert("RGB")

        # 构建 chat 消息
        messages = self._build_messages(pil_img, prompt)

        # 1) 构建文本模板
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2) 处理多模态输入（图像/视频）
        image_inputs, video_inputs = process_vision_info(messages)

        # 3) 组合为模型可接收的张量
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        # 4) 生成
        with self.gen_lock:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 与原逻辑一致，禁用采样
            )

        # 5) 剪去输入前缀，仅保留新增 token
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 6) 解码为文本
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_texts[0] if output_texts else ""

    def run_inference(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        try:
            ###
            prompt = prompt + '\nGive the answer choice and explain.'
            if not isinstance(image_path, str):
                raise ValueError("image_path must be a string.")
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            response = self._generate_response(image_path, prompt, max_new_tokens)
            return {"response": response}
        except Exception as e:
            trace_info = traceback.format_exc()
            print(f"Error during Lingshu VQA inference for '{image_path}' with prompt '{prompt}': {str(e)}\n{trace_info}")
            return {"error": f"{str(e)}"}


app = FastAPI()

class VQARequest(BaseModel):
    image_path: str
    prompt: str
    max_new_tokens: int = 150


lingshu_engine = LingshuServerEngine()

@app.post("/lingshu")
def vqa(request: VQARequest):
    result = lingshu_engine.run_inference(
        image_path=request.image_path,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lingshu-7B FastAPI Server")
    parser.add_argument(
        "--port",
        type=int,
        default=5007,
        help="Port to run the server on (default: 5007)"
    )
    args = parser.parse_args()

    print(f"Starting Lingshu-7B Server on 0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)