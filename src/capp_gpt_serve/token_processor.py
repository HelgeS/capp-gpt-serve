"""Token processing service for converting between JSON and token sequences."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

import torch

logger = logging.getLogger(__name__)


class TokenProcessor:
    """Service for handling token mapping and sequence conversion."""

    def __init__(
        self, token_mappings_path: Path, part_encoding_path: Optional[Path] = None
    ):
        """Initialize with token mappings from JSON file."""
        self.token_mappings_path = token_mappings_path
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[str, str] = {}

        # Load part encoding and process list
        if part_encoding_path is None:
            part_encoding_path = (
                Path(__file__).parent / "data" / "part_encoding_and_process_list.json"
            )
        self.part_encoding_path = part_encoding_path
        self.part_encoding: Dict[str, List[str]] = {}
        self.process_list: List[str] = []
        self.part_characteristic_tokens: Set[str] = set()

        self._load_mappings()
        self._load_part_encoding()

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

    def _load_part_encoding(self) -> None:
        """Load part encoding and process list from JSON file."""
        try:
            with open(self.part_encoding_path, "r") as f:
                data = json.load(f)

            self.part_encoding = data["part_encoding"]
            self.process_list = data["process_list"]

            logger.info(
                f"Loaded part encoding with {len(self.part_encoding)} categories and {len(self.process_list)} processes"
            )

            part_characteristic_tokens = set()
            for category, values in self.part_encoding.items():
                for value in values:
                    part_characteristic_tokens.add(f"{category}_{value}")
            self.part_characteristic_tokens = part_characteristic_tokens
            logger.info("Build set of part characteristic tokens to skip")

        except Exception as e:
            logger.error(f"Failed to load part encoding: {e}")
            raise

    def json_to_sequence(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Convert JSON input to token sequence."""
        try:
            sequence = []

            part_chars = input_data.get("part_characteristics", {})

            # Add tokens in order
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

            # Add start of sequence to prompt model for completion
            sequence.append(self.token2id["<SOS>"])

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

            # Find manufacturing process tokens (tokens after <SOS>)
            try:
                eoc_idx = token_ids.index(self.token2id["<SOS>"])
                process_tokens = token_ids[eoc_idx + 1 :]
            except ValueError:
                logger.warning(
                    "No <EOC> token found, treating entire sequence as processes"
                )
                process_tokens = token_ids

            # Convert process token IDs to names
            process_chains = []
            confidence_scores = []
            current_chain = []
            current_chain_confidence = []

            for token_id in process_tokens:
                token_str = str(token_id)
                if token_str in self.id2token:
                    token_name = self.id2token[token_str]

                    # Check if current process chain is complete
                    if token_name in ("<EOC>", "<EOS>"):
                        if len(current_chain) > 0:
                            process_chains.append(current_chain)
                            confidence_scores.append(current_chain_confidence)

                        current_chain = []
                        current_chain_confidence = []
                        continue

                    # Skip part characteristic tokens and all other special tokens
                    if (
                        token_name in self.part_characteristic_tokens
                        or token_name.startswith("<")
                    ):
                        continue

                    current_chain.append(token_name)

                    # Placeholder confidence score (would need actual model probabilities)
                    current_chain_confidence.append(0.8)

            result = {"process_chains": process_chains}

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
        categories = {}

        # Initialize categories from part encoding
        for category in self.part_encoding.keys():
            categories[category] = []

        # Add process_chains category
        categories["process_chains"] = []

        # Build set of part characteristic tokens
        part_characteristic_tokens = set()
        for category, values in self.part_encoding.items():
            for value in values:
                part_characteristic_tokens.add(f"{category}_{value}")

        # Categorize all tokens
        for token in self.token2id.keys():
            # Skip special tokens
            if token.startswith("<"):
                continue

            # Check if it's a part characteristic token
            categorized = False
            for category in self.part_encoding.keys():
                if token.startswith(f"{category}_"):
                    categories[category].append(token)
                    categorized = True
                    break

            # If not a part characteristic token, it's likely a process
            if not categorized:
                categories["process_chains"].append(token)

        return categories
