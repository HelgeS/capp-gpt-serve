# CAPP GPT Serve

HTTP API for GPT-2 based manufacturing process planning. This service provides REST endpoints to predict manufacturing processes based on part characteristics using a trained GPT-2 transformer model, optimized for performance with ONNX Runtime.

## Features

- **FastAPI-based REST API** with automatic OpenAPI documentation
- **Optimized GPT-2 model inference** using ONNX Runtime
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

By default, this project is configured to use a CPU-only version of PyTorch and ONNX Runtime to ensure compatibility across different environments.

If you are setting up a local development environment and have a CUDA-enabled GPU, you can install a version of ONNX Runtime with GPU support.

1.  **Follow the official ONNX Runtime installation instructions:** Visit the [ONNX Runtime website](https://onnxruntime.ai/docs/install/) to find the correct package for your system (e.g., `onnxruntime-gpu`).

2.  **Install ONNX Runtime with GPU support:**
    ```bash
    uv add onnxruntime-gpu
    ```
    This command will override the default CPU-only version.

3.  **Install other dependencies:** After installing the GPU-enabled version, you can proceed with installing the rest of the project's dependencies:
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
   uv run test-api
   ```

## API Endpoints
**Authentication**: If the `PREDICT_AUTH_TOKEN` environment variable is set, all `POST` requests to prediction endpoints (`/predict`, `/predict/explainable`) require an `X-Auth-Token` header with an access token: `X-Auth-Token: <your-token>`.

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
  "temperature": 1.0
}
```

**Response:**
```json
{
  "process_chains": [["Turning", "Milling", "Thread Milling"]],
  "process_confidence": [[0.95, 0.87, 0.73]],
  "chain_confidence": [0.60],
  "processing_time_ms": 45.2
}
```

### `GET /tokens`
Get all valid tokens organized by category.

### `GET /health`
Health check endpoint returning service status.

## Project Structure

```
├── data/
│   └── parts_and_process_chains.json # Training data
├── model/                   # GPT-2 ONNX model files
│   ├── config.json
│   ├── generation_config.json
│   ├── model.onnx
│   └── token_mappings.json
├── src/
│   ├── capp_gpt_serve/      # Main application package
│   │   ├── main.py          # FastAPI application
│   │   ├── model_service.py # Model loading and inference
│   │   ├── schemas.py       # Pydantic models
│   │   └── token_processor.py # Token mapping and conversion
│   └── scripts/             # Utility and test scripts
│       ├── benchmark_api.py
│       └── test_api.py
├── pyproject.toml           # Project configuration and dependencies
├── Dockerfile               # Multi-stage Docker build
└── docker-compose.yml       # Container orchestration
```

## Model Details

The service uses a custom GPT-2 model trained for manufacturing process planning, converted to ONNX for high-performance inference.
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
```

### Running Scripts

The project includes several scripts defined in `pyproject.toml` that can be executed with `uv run`.

```bash
# Run the API test script
uv run test-api

# Run the benchmark script
uv run benchmark

# Run the validation script
uv run validate
```

### Adding Dependencies
```bash
# Add runtime dependency
uv pip install package-name

# Add development dependency
uv pip install --dev package-name
```

## Production Deployment

1. **Environment variables:**
   - `PYTHONPATH`: Set to `/app`
   - `PORT`: API port (default: 8000)

2. **Resource requirements:**
   - Memory: 2GB+
   - CPU: 1+ cores
   - Disk: 1GB+ for model and dependencies

3. **Health monitoring:**
   - Health endpoint: `/health`
   - Docker health check included

## License

MIT License - see LICENSE file for details.
