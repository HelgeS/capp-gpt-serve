"""Pydantic schemas for API request and response validation."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


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
    max_processes: Optional[int] = Field(
        default=10, ge=1, le=20, description="Maximum number of processes to return"
    )
    temperature: Optional[float] = Field(
        default=1.0, ge=0.1, le=2.0, description="Sampling temperature"
    )
    include_confidence: Optional[bool] = Field(
        default=True, description="Include confidence scores in response"
    )


class InferenceResponse(BaseModel):
    """Response model for manufacturing process inference."""

    process_chains: List[str] = Field(
        ..., description="Recommended manufacturing processes"
    )
    confidence_scores: Optional[List[float]] = Field(
        None, description="Confidence scores for each process"
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


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
