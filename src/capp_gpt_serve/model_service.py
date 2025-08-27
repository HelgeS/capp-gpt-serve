"""Model service for loading and running inference with the GPT-2 manufacturing planning model."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Config
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton service for loading and running inference with the GPT-2 model."""
    
    _instance: Optional['ModelService'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ModelService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.model: Optional[GPT2LMHeadModel] = None
            self.config: Optional[GPT2Config] = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._initialized = True
    
    def load_model(self, model_path: Path) -> None:
        """Load the GPT-2 model from the specified path."""
        try:
            # Load configuration
            config_path = model_path / "config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            self.config = GPT2Config.from_dict(config_dict)
            logger.info(f"Loaded model config: {self.config}")
            
            # Initialize model
            self.model = GPT2LMHeadModel(self.config)
            
            # Load weights from safetensors
            safetensors_path = model_path / "model.safetensors"
            state_dict = load_file(str(safetensors_path))
            
            # Handle missing lm_head.weight by tying with transformer.wte.weight
            if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
                logger.info("Tying lm_head.weight with transformer.wte.weight")
                state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
            
            self.model.load_state_dict(state_dict, strict=False)
            
            # Move to device and set eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_sequence(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate a sequence using the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                input_ids = input_ids.to(self.device)
                
                # Set pad_token_id from config if not provided
                if pad_token_id is None:
                    pad_token_id = getattr(self.config, 'pad_token_id', 52)
                
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=getattr(self.config, 'eos_token_id', 51)
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