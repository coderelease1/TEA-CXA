import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple, Optional

import uuid
import tempfile
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import traceback
import argparse
from threading import Lock

class XRayPhraseGroundingEngine:
    DEFAULT_MODEL_PATH = "microsoft/maira-2"
    DEFAULT_DEVICE = "cuda:0"

    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: Optional[str] = None,
    ):
        self.model_path = self.DEFAULT_MODEL_PATH
        self.device = torch.device(self.DEFAULT_DEVICE)

        # Setup quantization config
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None

        # Load model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device, 
            cache_dir=cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # temp directory for visualization output
        #self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        #self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.gen_lock = Lock()
        print(f"Phrase Grounding model '{self.model_path}' loaded on device: {self.device}")
        #print(f"Visualization output dir: {self.temp_dir}")

    def _visualize_bboxes(
        self, image: Image.Image, bboxes: List[Tuple[float, float, float, float]], phrase: str
    ) -> str:
        plt.figure(figsize=(12, 12))
        plt.imshow(image, cmap="gray")

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            plt.gca().add_patch(
                plt.Rectangle(
                    (x1 * image.width, y1 * image.height),
                    width * image.width,
                    height * image.height,
                    fill=False,
                    color="red",
                    linewidth=2,
                )
            )

        plt.title(f"Located: {phrase}", pad=20)
        plt.axis("off")

        viz_path = self.temp_dir / f"grounding_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(viz_path, bbox_inches="tight", dpi=150)
        plt.close()

        return str(viz_path)

    def run_inference(self, image_path: str, phrase: str, max_new_tokens: int = 300) -> Dict[str, Any]:
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=image, phrase=phrase, return_tensors="pt"
            )
            # place tensor on device
            #inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            inputs = inputs.to(self.device)

            with self.gen_lock:
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )

            prompt_length = inputs["input_ids"].shape[-1]
            decoded_text = self.processor.decode(output[0][prompt_length:], skip_special_tokens=True)
            predictions_raw = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

            metadata = {
            }

            if not predictions_raw:
                return {
                    "predictions": [],
                    #"visualization_path": None,
                    "metadata": {**metadata, "analysis_status": "completed_no_finding"},
                }

            processed_predictions = []
            for pred_phrase, pred_bboxes in predictions_raw:
                if not pred_bboxes:
                    continue

                # Convert model bboxes to list format and get original image bboxes
                model_bboxes = [list(map(float, bbox)) for bbox in pred_bboxes]
                original_bboxes = [
                    self.processor.adjust_box_for_original_image_size(
                        bbox, width=image.size[0], height=image.size[1]
                    )
                    for bbox in model_bboxes
                ]

                processed_predictions.append(
                    {
                        "phrase": pred_phrase,
                        "bounding_boxes": {
                            "relative_coordinates": model_bboxes, #"model_coordinates"         # 相对坐标 [0,1]
                            #"image_coordinates": [                       # 绝对像素坐标 [x1, y1, x2, y2]
                            #    [float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in original_bboxes
                            #],
                        },
                    }
                )

            if processed_predictions:
                """ all_abs_bboxes = []
                for pred in processed_predictions:
                    all_abs_bboxes.extend(pred["bounding_boxes"]["image_coordinates"])
                viz_path = self._visualize_bboxes(image, all_abs_bboxes, phrase) """
            else:
                viz_path = None
                metadata["analysis_status"] = "completed_no_finding"

            return {
                "predictions": processed_predictions,
                #"visualization_path": viz_path,
                "metadata": metadata,
            }

        except Exception as e:
            trace_info = traceback.format_exc()
            print(f"Error during phrase grounding for '{image_path}' with phrase '{phrase}': {str(e)}\n{trace_info}")
            return {
                "error": f"{str(e)}",
            }


# ================= FastAPI Server =================

app = FastAPI()

class PhraseGroundingRequest(BaseModel):
    image_path: str = Field(..., description="Path to the frontal chest X-ray image (JPG/PNG)")
    phrase: str = Field(..., description="Medical phrase to ground (e.g., 'Pleural effusion')")
    max_new_tokens: int = Field(150, description="Maximum number of new tokens to generate")

PhraseGrounding_engine = XRayPhraseGroundingEngine(cache_dir="/your/path/model-weights",load_in_8bit=True)#load_in_8bit=True, 

@app.post("/phrase_grounding")
def phrase_grounding(request: PhraseGroundingRequest):
    result = PhraseGrounding_engine.run_inference(
        image_path=request.image_path,
        phrase=request.phrase,
        max_new_tokens=request.max_new_tokens,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGemma FastAPI Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5006, 
        help="Port to run the server on (default: 5007)"
    )

    args = parser.parse_args()

    # Launch the server on http://0.0.0.0:5006
    print(f"Starting X-Ray Phrase Grounding Server on 0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)