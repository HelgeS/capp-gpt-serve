"""Explainability service for processing model attention weights and scores."""

import logging
from typing import Dict, List, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ExplainabilityProcessor:
    """Processor for generating explainability information from model outputs."""

    def __init__(self):
        """Initialize the explainability processor."""
        self.input_characteristic_keys = [
            "geometry",
            "holes",
            "external_threads",
            "surface_finish",
            "tolerance",
            "batch_size",
        ]

    def process_explainability_data(
        self,
        input_sequence: torch.Tensor,
        attentions: List[torch.Tensor],
        process_chains: List[List[str]],
        input_characteristics: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Process model outputs to generate explainability information.

        Args:
            input_sequence: Original input token sequence
            scores: Token scores from model generation
            attentions: Attention weights from model generation
            process_chains: Decoded process chains
            input_characteristics: Original input characteristics

        Returns:
            Dictionary containing explainability data
        """

        try:
            # Calculate input influences from attention weights
            input_influences = self._calculate_input_influences(
                input_sequence, attentions, process_chains, input_characteristics
            )

            return {"input_influences": input_influences}

        except Exception as e:
            logger.error(f"Failed to process explainability data: {e}")
            # Return default values if processing fails
            return {"input_influences": [{} for _ in process_chains]}

    def _calculate_input_influences(
        self,
        input_sequence: torch.Tensor,
        attentions: List[torch.Tensor],
        process_chains: List[List[str]],
        input_characteristics: Dict[str, str],
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Calculate input influence mapping for each process in each chain.

        Args:
            input_sequence: Original input token sequence
            attentions: Attention weights from model generation
            process_chains: Decoded process chains
            input_characteristics: Original input characteristics

        Returns:
            List of influence mappings per chain
        """
        influences_per_chain = []
        input_length = input_sequence.size(1)  # Number of input tokens

        # We expect input_length to be 7 (6 characteristics + 1 SOS token)
        # The first 6 tokens correspond to the input characteristics
        characteristic_positions = list(
            range(min(6, input_length - 1))
        )  # Exclude SOS token

        for chain_idx, chain in enumerate(process_chains):
            chain_influences = {}

            for process_step, process in enumerate(chain):
                # Get attention weights for this process step
                if process_step < len(attentions):
                    # attentions[step] is tuple of layers: (layer_0, layer_1, ..., layer_n)
                    step_attentions = attentions[process_step]

                    if step_attentions and len(step_attentions) > 0:
                        # Average attention across all layers and heads
                        avg_attention_to_input = self._average_attention_to_input(
                            step_attentions, characteristic_positions
                        )

                        # Map averaged attention to characteristic names
                        process_influences = {}
                        for i, char_name in enumerate(self.input_characteristic_keys):
                            if i < len(avg_attention_to_input):
                                process_influences[char_name] = avg_attention_to_input[
                                    i
                                ]
                            else:
                                process_influences[char_name] = 0.0

                        # Normalize to sum to 1.0
                        total = sum(process_influences.values())
                        if total > 0:
                            for key in process_influences:
                                process_influences[key] = (
                                    process_influences[key] / total
                                )
                    else:
                        # Fallback to uniform distribution
                        process_influences = self._uniform_influences()
                else:
                    # No attention data for this step, use uniform
                    process_influences = self._uniform_influences()

                chain_influences[process] = process_influences

            influences_per_chain.append(chain_influences)

        return influences_per_chain

    def _average_attention_to_input(
        self, step_attentions: tuple, characteristic_positions: List[int]
    ) -> List[float]:
        """
        Average attention weights across layers and heads for input positions.

        Args:
            step_attentions: Tuple of attention tensors for each layer
            characteristic_positions: List of input token positions to analyze

        Returns:
            List of averaged attention weights for each characteristic position
        """
        if not step_attentions or len(step_attentions) == 0:
            return [0.0] * len(characteristic_positions)

        accumulated_attention = [0.0] * len(characteristic_positions)
        valid_layers = 0

        for layer_attention in step_attentions:
            if layer_attention is not None:
                # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                if layer_attention.dim() == 4:
                    # Average across batch and heads: [seq_len, seq_len]
                    avg_layer_attention = layer_attention.mean(dim=(0, 1))

                    # Get attention from the last position (current generation step) to input positions
                    current_pos = avg_layer_attention.size(0) - 1

                    for i, pos in enumerate(characteristic_positions):
                        if pos < avg_layer_attention.size(1):
                            accumulated_attention[i] += avg_layer_attention[
                                current_pos, pos
                            ].item()

                    valid_layers += 1

        # Average across layers
        if valid_layers > 0:
            return [att / valid_layers for att in accumulated_attention]
        else:
            return [0.0] * len(characteristic_positions)

    def _uniform_influences(self) -> Dict[str, float]:
        """Return uniform influence distribution."""
        uniform_weight = 1.0 / len(self.input_characteristic_keys)
        return {key: uniform_weight for key in self.input_characteristic_keys}


# Global instance
explainability_processor = ExplainabilityProcessor()
