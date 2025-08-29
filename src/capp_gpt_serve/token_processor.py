"""Token processing service for converting between JSON and token sequences."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

logger = logging.getLogger(__name__)


class TokenProcessor:
    """Service for handling token mapping and sequence conversion."""

    def __init__(self, token_mappings_path: Path):
        """Initialize with token mappings from JSON file."""
        self.token_mappings_path = token_mappings_path
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[str, str] = {}
        self._load_mappings()

    def _load_mappings(self) -> None:
        """Load token mappings from JSON file."""
        try:
            with open(self.token_mappings_path, "r") as f:
                mappings = json.load(f)

            self.token2id = mappings["token2id"]
            self.id2token = mappings["id2token"]

            logger.info(f"Loaded {len(self.token2id)} token mappings")

        except Exception as e:
            logger.error(f"Failed to load token mappings: {e}")
            raise

    def json_to_sequence(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Convert JSON input to token sequence."""
        try:
            # Start with SOS token
            sequence = [self.token2id["<SOS>"]]

            # Process part characteristics
            part_chars = input_data.get("part_characteristics", {})

            # Add tokens in order: geometry, holes, external_threads, surface_finish, tolerance, batch_size
            for key in [
                "geometry",
                "holes",
                "external_threads",
                "surface_finish",
                "tolerance",
                "batch_size",
            ]:
                if key in part_chars:
                    value = part_chars[key]
                    if value in self.token2id:
                        sequence.append(self.token2id[value])
                    else:
                        logger.warning(f"Unknown token: {value}")

            # Add end of context token
            sequence.append(self.token2id["<EOC>"])

            return torch.tensor([sequence], dtype=torch.long)

        except Exception as e:
            logger.error(f"Failed to convert JSON to sequence: {e}")
            raise

    def sequence_to_json(
        self, sequence: torch.Tensor, include_confidence: bool = True
    ) -> Dict[str, Any]:
        """Convert token sequence back to JSON format."""
        try:
            # Convert tensor to list and remove batch dimension
            if sequence.dim() > 1:
                sequence = sequence.squeeze(0)
            token_ids = sequence.tolist()

            # Find manufacturing processes (tokens after <EOC>)
            processes = []
            confidence_scores = []

            try:
                eoc_idx = token_ids.index(self.token2id["<EOC>"])
                process_tokens = token_ids[eoc_idx + 1 :]
            except ValueError:
                logger.warning(
                    "No <EOC> token found, treating entire sequence as processes"
                )
                process_tokens = token_ids

            # Convert process token IDs to names
            for token_id in process_tokens:
                token_str = str(token_id)
                if token_str in self.id2token:
                    token_name = self.id2token[token_str]
                    # Skip special tokens
                    if not token_name.startswith("<") and token_name not in [
                        "geometry_pure_axisymmetric",
                        "geometry_axisymmetric_with_prismatic_features",
                        "geometry_prismatic",
                        "geometry_prismatic_with_axisymmetric_features",
                        "geometry_prismatic_with_freeform_features",
                        "geometry_freeform",
                        "geometry_unconventional",
                        "holes_none",
                        "holes_normal",
                        "holes_normal_threaded",
                        "holes_normal_functional",
                        "holes_large",
                        "holes_large_threaded",
                        "holes_large_functional",
                        "external_threads_yes",
                        "external_threads_no",
                        "surface_finish_rough",
                        "surface_finish_normal",
                        "surface_finish_good",
                        "surface_finish_very_good",
                        "tolerance_rough",
                        "tolerance_normal",
                        "tolerance_medium",
                        "tolerance_tight",
                        "batch_size_prototype",
                        "batch_size_small",
                        "batch_size_medium",
                        "batch_size_large",
                        "batch_size_mass",
                    ]:
                        processes.append(token_name)
                        # Placeholder confidence score (would need actual model probabilities)
                        confidence_scores.append(0.8)

            result = {"process_chains": processes}

            if include_confidence:
                result["confidence_scores"] = confidence_scores

            return result

        except Exception as e:
            logger.error(f"Failed to convert sequence to JSON: {e}")
            raise

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input JSON structure and tokens."""
        try:
            if "part_characteristics" not in input_data:
                return False

            part_chars = input_data["part_characteristics"]

            # Check if all values are valid tokens
            for key, value in part_chars.items():
                if value not in self.token2id:
                    logger.error(f"Invalid token: {value}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False

    def get_valid_tokens(self) -> Dict[str, List[str]]:
        """Get all valid tokens organized by category."""
        categories = {
            "geometry": [],
            "holes": [],
            "external_threads": [],
            "surface_finish": [],
            "tolerance": [],
            "batch_size": [],
            "process_chains": [],
        }

        for token in self.token2id.keys():
            if token.startswith("geometry_"):
                categories["geometry"].append(token)
            elif token.startswith("holes_"):
                categories["holes"].append(token)
            elif token.startswith("external_threads_"):
                categories["external_threads"].append(token)
            elif token.startswith("surface_finish_"):
                categories["surface_finish"].append(token)
            elif token.startswith("tolerance_"):
                categories["tolerance"].append(token)
            elif token.startswith("batch_size_"):
                categories["batch_size"].append(token)
            elif not token.startswith("<") and token not in sum(
                categories.values(), []
            ):
                categories["process_chains"].append(token)

        return categories
