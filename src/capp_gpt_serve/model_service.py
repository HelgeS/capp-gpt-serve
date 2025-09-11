"""Model service for loading and running inference with the GPT-2 manufacturing planning model."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from optimum.onnxruntime import ORTModelForCausalLM

import torch
from transformers import GPT2Config, GPT2LMHeadModel

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
            self.pytorch_model: Optional[GPT2LMHeadModel] = None
            self.config: Optional[GPT2Config] = None
            # CPU-only for now, otherwise use: torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = "cpu"
            self._initialized = True
            self.pad_token_id: int = 52  # Default pad token ID
            self.eos_token_id: int = 51  # Default EOS token ID
            self._original_model_path: Optional[Path] = None

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

            # Initialize ONNX model for fast inference
            self.model = ORTModelForCausalLM.from_pretrained(model_path)

            # Load PyTorch model for attention extraction
            info_file = model_path / "info.txt"
            if info_file.exists():
                with open(info_file, "r") as f:
                    original_path = f.read().strip()
                    self._original_model_path = Path(original_path)

                    if self._original_model_path.exists():
                        logger.info(
                            f"Loading PyTorch model for attention extraction: {self._original_model_path}"
                        )
                        self.pytorch_model = GPT2LMHeadModel.from_pretrained(
                            self._original_model_path,
                            local_files_only=True,
                            attn_implementation="eager",  # Required for attention output
                        )
                        self.pytorch_model.to(self.device)
                        self.pytorch_model.eval()
                        logger.info(
                            "PyTorch model loaded successfully for attention extraction"
                        )
                    else:
                        logger.warning(
                            f"Original PyTorch model not found at: {self._original_model_path}"
                        )
            else:
                logger.warning(
                    "No info.txt file found - attention extraction will not be available"
                )

            # Quantize, if needed
            # if False:
            #     quantizer = ORTQuantizer.from_pretrained(onnx_path)
            #     dqconfig = AutoQuantizationConfig.avx2(
            #         is_static=False, per_channel=False
            #     )
            #     quantizer.quantize(
            #         save_dir="gpt2_quantize.onnx", quantization_config=dqconfig
            #     )
            #     self.model = ORTModelForCausalLM.from_pretrained("gpt2_quantize.onnx")

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
                    return_dict_in_generate=True,
                    output_logits=True,
                )
                return outputs

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_sequence_with_explainability(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Generate a sequence with explainability information using hybrid approach."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            with torch.no_grad():
                input_ids = input_ids.to(self.device)

                try:
                    logger.debug("Extracting attention weights using PyTorch model")
                    result = self.pytorch_model.generate(
                        input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_p=top_p,
                        pad_token_id=self.pad_token_id,
                        eos_token_id=self.eos_token_id,
                        output_attentions=True,
                        output_logits=True,
                        return_dict_in_generate=True,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract attention weights from PyTorch model: {e}"
                    )
                    result = {}

                return result

        except Exception as e:
            logger.error(f"Generation with explainability failed: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.config is not None


# Global instance
model_service = ModelService()
