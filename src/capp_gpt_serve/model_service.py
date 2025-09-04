"""Model service for loading and running inference with the GPT-2 manufacturing planning model."""

import json
import logging
from pathlib import Path
from typing import Optional

from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

import torch
from transformers import GPT2Config

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton service for loading and running inference with the GPT-2 model."""

    _instance: Optional["ModelService"] = None
    _initialized: bool = False

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model: Optional[ORTModelForCausalLM] = None
            self.config: Optional[GPT2Config] = None
            # CPU-only for now, otherwise use: torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = "cpu"
            self._initialized = True
            self.pad_token_id: int = 52  # Default pad token ID
            self.eos_token_id: int = 51  # Default EOS token ID

    def load_model(self, model_path: Path) -> None:
        """Load the GPT-2 model from the specified path."""
        try:
            # Load configuration
            config_path = model_path / "config.json"
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            self.config = GPT2Config.from_dict(config_dict)
            logger.info(f"Loaded model config: {self.config}")

            self.pad_token_id = getattr(self.config, "pad_token_id", self.pad_token_id)
            logger.info(f"Pad token ID: {self.pad_token_id}")

            self.eos_token_id = getattr(self.config, "eos_token_id", self.eos_token_id)
            logger.info(f"EOS token ID: {self.eos_token_id}")

            # Initialize model
            # self.model = GPT2LMHeadModel.from_pretrained(
            #    model_path, local_files_only=True
            # )
            self.model = ORTModelForCausalLM.from_pretrained(model_path)

            # Quantize, if needed
            if False:
                quantizer = ORTQuantizer.from_pretrained(onnx_path)
                dqconfig = AutoQuantizationConfig.avx2(
                    is_static=False, per_channel=False
                )
                quantizer.quantize(
                    save_dir="gpt2_quantize.onnx", quantization_config=dqconfig
                )
                self.model = ORTModelForCausalLM.from_pretrained("gpt2_quantize.onnx")

            # Move to device and set eval mode
            self.model.to(self.device)
            #            self.model.eval()

            logger.info(f"Model loaded successfully on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate a sequence using the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            with torch.no_grad():
                input_ids = input_ids.to(self.device)

                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )
                return outputs

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.config is not None


# Global instance
model_service = ModelService()
