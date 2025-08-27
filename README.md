# CAPP GPT Serve

HTTP API for GPT-2 based manufacturing process planning. This service provides REST endpoints to predict manufacturing processes based on part characteristics using a trained GPT-2 transformer model.

## Features

- **FastAPI-based REST API** with automatic OpenAPI documentation
- **GPT-2 model inference** for manufacturing process planning
- **JSON input/output** with Pydantic validation
- **Docker support** with multi-stage builds using uv
- **Health monitoring** and error handling
- **Token validation** and category listing

## Quick Start

### Using uv (Recommended)

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the development server:**
   ```bash
   uv run serve
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Local Development with GPU Support

By default, this project is configured to use a CPU-only version of PyTorch to ensure compatibility across different environments, especially within the provided Docker container.

If you are setting up a local development environment and have a CUDA-enabled GPU, you can install a version of PyTorch with GPU support.

1.  **Follow the official PyTorch installation instructions:** Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and select the appropriate options for your system (e.g., Linux, Pip, Python, and your CUDA version).

2.  **Install PyTorch with GPU support:** Run the command provided by the PyTorch website. It will look something like this (example for CUDA 12.1):
    ```bash
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    This command will override the default CPU-only version specified in the project's configuration.

3.  **Install other dependencies:** After installing the GPU-enabled version of PyTorch, you can proceed with installing the rest of the project's dependencies:
    ```bash
    uv sync
    ```

### Using Docker

1. **Build and run:**
   ```bash
   docker-compose up --build
   ```

2. **Test the API:**
   ```bash
   python test_api.py
   ```

## API Endpoints

### `POST /predict`
Predict manufacturing processes for given part characteristics.

**Request:**
```json
{
  "part_characteristics": {
    "geometry": "geometry_prismatic",
    "holes": "holes_normal",
    "external_threads": "external_threads_yes",
    "surface_finish": "surface_finish_normal",
    "tolerance": "tolerance_medium",
    "batch_size": "batch_size_small"
  },
  "max_processes": 5,
  "temperature": 0.8,
  "include_confidence": true
}
```

**Response:**
```json
{
  "manufacturing_processes": ["Turning", "Milling", "Thread Milling"],
  "confidence_scores": [0.95, 0.87, 0.73],
  "processing_time_ms": 45.2
}
```

### `GET /tokens`
Get all valid tokens organized by category.

### `GET /health`
Health check endpoint returning service status.

## Project Structure

```
├── src/capp_gpt_serve/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── model_service.py     # Model loading and inference
│   ├── token_processor.py   # Token mapping and conversion
│   └── schemas.py           # Pydantic models
├── model/                   # GPT-2 model files
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   └── token_mappings.json
├── pyproject.toml          # Project configuration
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Container orchestration
└── test_api.py            # API test script
```

## Model Details

The service uses a custom GPT-2 model trained for manufacturing process planning:
- **Architecture:** GPT2LMHeadModel
- **Parameters:** 4 layers, 4 attention heads, 64 embedding dimensions
- **Vocabulary:** 53 tokens (manufacturing-specific)
- **Context length:** 512 tokens

## Development

### Code Quality
```bash
# Format code
uv run black .

# Lint code  
uv run ruff check .

# Run tests
uv run pytest
```

### Adding Dependencies
```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

## Production Deployment

1. **Environment variables:**
   - `PYTHONPATH`: Set to `/app/src`
   - `PORT`: API port (default: 8000)

2. **Resource requirements:**
   - Memory: 4GB+ (for PyTorch model)
   - CPU: 2+ cores recommended
   - Disk: 2GB+ for model and dependencies

3. **Health monitoring:**
   - Health endpoint: `/health`
   - Docker health check included
   - Prometheus metrics (optional)

## License

MIT License - see LICENSE file for details.
