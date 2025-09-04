"""Main FastAPI application for the manufacturing process planning API."""

import time
import logging
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .model_service import model_service
from .token_processor import TokenProcessor
from .schemas import (
    InferenceRequest,
    InferenceResponse,
    HealthResponse,
    TokenCategoriesResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CAPP GPT Serve",
    description="HTTP API for GPT-2 based manufacturing process planning",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global token processor
token_processor: TokenProcessor = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global token_processor

    try:
        # Initialize paths
        model_path = Path("model")
        token_mappings_path = model_path / "token_mappings.json"

        # Verify paths exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        if not token_mappings_path.exists():
            raise FileNotFoundError(f"Token mappings not found: {token_mappings_path}")

        # Initialize token processor
        token_processor = TokenProcessor(token_mappings_path)
        logger.info("Token processor initialized")

        # Load model
        model_service.load_model(model_path)
        logger.info("Model service initialized")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", model_loaded=model_service.is_loaded(), version="0.1.0"
    )


@app.get("/tokens", response_model=TokenCategoriesResponse)
async def get_valid_tokens():
    """Get all valid tokens organized by category."""
    if token_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token processor not initialized",
        )

    categories = token_processor.get_valid_tokens()
    return TokenCategoriesResponse(**categories)


@app.post("/predict", response_model=InferenceResponse)
async def predict_process_chains(request: InferenceRequest):
    """Predict manufacturing processes for given part characteristics."""
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    if token_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token processor not initialized",
        )

    start_time = time.time()

    try:
        # Convert request to dictionary for processing
        input_data = {"part_characteristics": request.part_characteristics.model_dump()}

        # Validate input
        if not token_processor.validate_input(input_data):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input tokens"
            )

        # Convert JSON to token sequence
        input_sequence = token_processor.json_to_sequence(input_data)

        # Generate output sequence
        output_sequence = model_service.generate_sequence(
            input_sequence,
            max_length=512, #input_sequence.size(1) + request.max_processes,
            temperature=request.temperature,
        )

        # Convert back to JSON
        result = token_processor.sequence_to_json(
            output_sequence, include_confidence=request.include_confidence
        )

        # Limit number of processes if requested
        if len(result["process_chains"]) > request.max_processes:
            result["process_chains"] = result["process_chains"][
                : request.max_processes
            ]
            if result.get("confidence_scores"):
                result["confidence_scores"] = result["confidence_scores"][
                    : request.max_processes
                ]

        processing_time = (time.time() - start_time) * 1000

        return InferenceResponse(
            process_chains=result["process_chains"],
            confidence_scores=result.get("confidence_scores"),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": "CAPP GPT Serve",
        "version": "0.1.0",
        "description": "HTTP API for GPT-2 based manufacturing process planning",
        "docs": "/docs",
        "health": "/health",
    }


def cli():
    """Command line interface for running the server."""
    uvicorn.run(
        "capp_gpt_serve.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
