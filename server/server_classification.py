import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional, Tuple, Any, Type

import skimage.io
import torch
import torchvision
import torchxrayvision as xrv
import traceback


class ChestXRayEngine:
    DEFAULT_MODEL_NAME = "densenet121-res224-all"
    DEFAULT_DEVICE = "cuda:0"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):

        self.model_name = self.DEFAULT_MODEL_NAME
        device_cfg = self.DEFAULT_DEVICE

        self.device = torch.device(device_cfg)

        self.model = xrv.models.DenseNet(weights=self.model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
        print(f"ChestXRay model '{self.model_name}' loaded on device: {self.device}")

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess the X-ray image for inference.
        Supports JPG/PNG.
        """
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)

        # for the case of [N, H, W] or [N, H, W, C]
        if img.ndim >= 4:
            img = img[0]
        elif img.ndim == 3 and img.shape[0] > 5 and img.shape[2] not in (1, 3, 4):
            # for the case of [N, H, W] rather than [H, W, C]
            img = img[0]
        
        # If RGB, take single channel
        if len(img.shape) > 2:
            img = img[:, :, 0]

        img = img[None, :, :]
        img = self.transform(img)  # XRayCenterCrop expects numpy HxW or 1xHxW
        img = torch.from_numpy(img).unsqueeze(0)  # shape: [1, 1, H, W]
        img = img.to(self.device)
        return img

    def run_inference(self, image_path: str) -> Dict[str, Any]:

        try:
            img = self._process_image(image_path)
            with torch.inference_mode():
                preds = self.model(img).detach().cpu()[0]

            rounded = [round(float(v), 3) for v in preds.numpy().tolist()]
            predictions = dict(zip(xrv.datasets.default_pathologies, rounded))
            return {
                "predictions": predictions,
            }
        except Exception as e:
            trace_info = traceback.format_exc()
            print(f"Error during X-ray inference for '{image_path}': {str(e)}\n{trace_info}")
            return {
                "error": f"{str(e)}",
            }


app = FastAPI()

class ClassificationRequest(BaseModel):
    image_path: str


engine = ChestXRayEngine()


@app.post("/classification")
def classification(request: ClassificationRequest):
    
    result = engine.run_inference(request.image_path)
    return result


if __name__ == "__main__":
    # Launch the server on http://0.0.0.0:5003
    print("Starting Chest X-Ray Classification Server on 0.0.0.0:5003")
    uvicorn.run(app, host="0.0.0.0", port=5003)