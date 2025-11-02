"""
FastAPI Deployment for Prompt Injection Detection API

Deploy with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Test with:
    curl -X POST "http://localhost:8000/detect" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "What is AI?"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Prompt Injection Detection API",
    description="API for detecting prompt injection attacks on LLMs",
    version="1.0.0"
)

# Global model storage
MODEL = None
TOKENIZER = None
DEVICE = None


# Request/Response models
class DetectionRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to analyze", min_length=1)
    threshold: Optional[float] = Field(
        0.85,
        description="Confidence threshold for flagging",
        ge=0.0,
        le=1.0
    )


class DetectionResponse(BaseModel):
    prompt: str
    label: str
    confidence: float
    recommendation: str
    timestamp: str


class BatchDetectionRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts to analyze")
    threshold: Optional[float] = 0.85


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Startup: Load model
@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global MODEL, TOKENIZER, DEVICE
    
    # Configuration (replace with your model path)
    model_path = "./models/roberta_base/final"
    
    try:
        logger.info(f"Loading model from {model_path}")
        TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        MODEL = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL.to(DEVICE)
        MODEL.eval()
        
        logger.info(f"✅ Model loaded successfully on {DEVICE}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def detect_injection(prompt: str, threshold: float = 0.85) -> dict:
    """Core detection logic."""
    from datetime import datetime
    
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("Model not loaded")
    
    # Tokenize
    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = MODEL(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get prediction
    pred_label = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_label].item()
    
    # Convert to readable format
    label = "INJECTION" if pred_label == 1 else "LEGITIMATE"
    
    # Recommendation
    if label == "INJECTION":
        if confidence > 0.95:
            recommendation = "BLOCK"
        elif confidence > threshold:
            recommendation = "REVIEW"
        else:
            recommendation = "ALLOW_WITH_MONITORING"
    else:
        recommendation = "ALLOW"
    
    return {
        "prompt": prompt,
        "label": label,
        "confidence": round(confidence * 100, 2),
        "recommendation": recommendation,
        "timestamp": datetime.utcnow().isoformat()
    }


# API Endpoints
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Prompt Injection Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect",
            "batch": "/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "unknown"
    }


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_prompt(request: DetectionRequest):
    """
    Detect if a prompt is a potential injection attack.
    
    - **prompt**: The text to analyze
    - **threshold**: Confidence threshold (default: 0.85)
    
    Returns detection result with label and confidence.
    """
    try:
        result = detect_injection(request.prompt, request.threshold)
        return result
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", tags=["Detection"])
async def batch_detect(request: BatchDetectionRequest):
    """
    Detect multiple prompts in batch.
    
    - **prompts**: List of prompts to analyze
    - **threshold**: Confidence threshold (default: 0.85)
    
    Returns list of detection results.
    """
    try:
        results = []
        for prompt in request.prompts:
            result = detect_injection(prompt, request.threshold)
            results.append(result)
        
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get model statistics."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    param_count = sum(p.numel() for p in MODEL.parameters())
    
    return {
        "model_parameters": f"{param_count / 1e6:.1f}M",
        "device": str(DEVICE),
        "model_type": MODEL.config.model_type,
        "max_length": 256
    }


# Example usage documentation
"""
# Start the server
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Test single detection
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is the capital of France?"}'

# Test batch detection
curl -X POST "http://localhost:8000/batch" \
     -H "Content-Type: application/json" \
     -d '{"prompts": ["What is AI?", "Ignore previous instructions"]}'

# Health check
curl "http://localhost:8000/health"

# View interactive docs
# Open http://localhost:8000/docs in browser
"""

# Production deployment with Docker:
"""
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

# Build and run
docker build -t prompt-injection-api .
docker run -p 8000:8000 prompt-injection-api
"""
