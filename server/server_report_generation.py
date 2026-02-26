import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional, Any

import torch
from PIL import Image
from transformers import (
    BertTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    GenerationConfig,
)
import traceback


class ChestXRayReportEngine:
    DEFAULT_DEVICE = "cuda:2"
    FINDINGS_MODEL_ID = "IAMJB/chexpert-mimic-cxr-findings-baseline"
    IMPRESSION_MODEL_ID = "IAMJB/chexpert-mimic-cxr-impression-baseline"

    def __init__(self, cache_dir: str = None):
        # 设备配置
        device_cfg = self.DEFAULT_DEVICE
        self.device = torch.device(device_cfg)

        print("Loading findings model and processors...")
        self.findings_model = VisionEncoderDecoderModel.from_pretrained(
            self.FINDINGS_MODEL_ID, cache_dir=cache_dir
        ).eval().to(self.device)
        self.findings_tokenizer = BertTokenizer.from_pretrained(
            self.FINDINGS_MODEL_ID, cache_dir=cache_dir
        )
        self.findings_processor = ViTImageProcessor.from_pretrained(
            self.FINDINGS_MODEL_ID, cache_dir=cache_dir
        )

        print("Loading impression model and processors...")
        self.impression_model = VisionEncoderDecoderModel.from_pretrained(
            self.IMPRESSION_MODEL_ID, cache_dir=cache_dir
        ).eval().to(self.device)
        self.impression_tokenizer = BertTokenizer.from_pretrained(
            self.IMPRESSION_MODEL_ID, cache_dir=cache_dir
        )
        self.impression_processor = ViTImageProcessor.from_pretrained(
            self.IMPRESSION_MODEL_ID, cache_dir=cache_dir
        )

        self.generation_args = {
            "num_return_sequences": 1,
            "max_length": 128,
            "use_cache": True,
            "num_beams": 2,
        }


    def _process_image(self, image_path: str, processor: ViTImageProcessor, model: VisionEncoderDecoderModel) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values

        expected_size = getattr(model.config.encoder, "image_size", None)
        if expected_size is not None:
            # support int or (h, w)
            if isinstance(expected_size, int):
                target_size = (expected_size, expected_size)
            elif isinstance(expected_size, (tuple, list)) and len(expected_size) == 2:
                target_size = tuple(expected_size)

            if (pixel_values.shape[-2], pixel_values.shape[-1]) != target_size:
                pixel_values = torch.nn.functional.interpolate(
                    pixel_values, size=target_size, mode="bilinear", align_corners=False
                )

        pixel_values = pixel_values.to(self.device)
        return pixel_values

    def _generate_text(self, pixel_values: torch.Tensor, model: VisionEncoderDecoderModel, tokenizer: BertTokenizer) -> str:
        generation_config = GenerationConfig(
            **{
                **self.generation_args,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "decoder_start_token_id": tokenizer.cls_token_id,
            }
        )
        generated_ids = model.generate(pixel_values, generation_config=generation_config)
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def generate_report(self, image_path: str) -> Dict[str, Any]:
        try:
            # Process image for both models
            findings_pixels = self._process_image(image_path, self.findings_processor, self.findings_model)
            impression_pixels = self._process_image(image_path, self.impression_processor, self.impression_model)

            # Generate both sections
            with torch.inference_mode():
                findings_text = self._generate_text(findings_pixels, self.findings_model, self.findings_tokenizer)
                impression_text = self._generate_text(impression_pixels, self.impression_model, self.impression_tokenizer)

            # Combine into formatted report
            report_text = (
                "CHEST X-RAY REPORT\n\n"
                f"FINDINGS:\n{findings_text}\n\n"
                f"IMPRESSION:\n{impression_text}"
            )

            return {
                "report": report_text,
            }
        except Exception as e:
            trace_info = traceback.format_exc()
            print(f"Error during report generation for '{image_path}': {str(e)}\n{trace_info}")
            return {
                "error": f"{str(e)}",
            }


app = FastAPI()

class ReportRequest(BaseModel):
    image_path: str


report_engine = ChestXRayReportEngine(cache_dir="/your/path/model-weights")

@app.post("/report_generation")
def report_generation(request: ReportRequest):
    result = report_engine.generate_report(request.image_path)
    return result


if __name__ == "__main__":
    print("Starting Chest X-Ray Report Generation Server on 0.0.0.0:5005")
    uvicorn.run(app, host="0.0.0.0", port=5005)