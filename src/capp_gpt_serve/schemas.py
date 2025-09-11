"""Pydantic schemas for API request and response validation."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class PartCharacteristics(BaseModel):
    """Model for part characteristics input."""

    geometry: str = Field(..., description="Geometry type of the part")
    holes: str = Field(..., description="Type of holes in the part")
    external_threads: str = Field(..., description="Whether part has external threads")
    surface_finish: str = Field(..., description="Required surface finish quality")
    tolerance: str = Field(..., description="Required tolerance level")
    batch_size: str = Field(..., description="Production batch size")


class InferenceRequest(BaseModel):
    """Request model for manufacturing process inference."""

    part_characteristics: PartCharacteristics
    temperature: Optional[float] = Field(
        default=1.0, ge=0.1, le=2.0, description="Sampling temperature"
    )


class InferenceResponse(BaseModel):
    """Response model for manufacturing process inference."""

    process_chains: List[List[str]] = Field(
        ..., description="Recommended manufacturing processes"
    )
    process_confidence: List[List[float]] = Field(
        ..., description="Confidence scores for each process"
    )
    chain_confidence: List[float] = Field(
        ...,
        description="Overall confidence score for each process chain (0-1, higher is more confident)",
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class TokenCategoriesResponse(BaseModel):
    """Response model for valid token categories."""

    geometry: List[str] = Field(..., description="Valid geometry tokens")
    holes: List[str] = Field(..., description="Valid hole tokens")
    external_threads: List[str] = Field(..., description="Valid external thread tokens")
    surface_finish: List[str] = Field(..., description="Valid surface finish tokens")
    tolerance: List[str] = Field(..., description="Valid tolerance tokens")
    batch_size: List[str] = Field(..., description="Valid batch size tokens")
    process_chains: List[str] = Field(
        ..., description="Possible manufacturing processes"
    )


class InputInfluence(BaseModel):
    """Model for input influence mapping per process step."""

    geometry: Optional[float] = Field(
        None, description="Influence of geometry on this process"
    )
    holes: Optional[float] = Field(
        None, description="Influence of holes on this process"
    )
    external_threads: Optional[float] = Field(
        None, description="Influence of external threads on this process"
    )
    surface_finish: Optional[float] = Field(
        None, description="Influence of surface finish on this process"
    )
    tolerance: Optional[float] = Field(
        None, description="Influence of tolerance on this process"
    )
    batch_size: Optional[float] = Field(
        None, description="Influence of batch size on this process"
    )


class ExplainabilityData(BaseModel):
    """Model for explainability information."""

    input_influences: List[Dict[str, Dict[str, float]]] = Field(
        ..., description="Input influence mapping for each process in each chain"
    )


class ExplainableInferenceRequest(BaseModel):
    """Request model for explainable manufacturing process inference."""

    part_characteristics: PartCharacteristics
    temperature: Optional[float] = Field(
        default=1.0, ge=0.1, le=2.0, description="Sampling temperature"
    )


class ExplainableInferenceResponse(InferenceResponse):
    """Response model for explainable manufacturing process inference."""

    explainability: ExplainabilityData = Field(
        ...,
        description="Explainability information including uncertainty and input influences",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
