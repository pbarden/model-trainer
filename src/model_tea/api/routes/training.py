from fastapi import APIRouter, HTTPException
from model_tea.api.schemas import (
    TrainModelRequest,
    TrainDirectRequest,
    TrainingStatusResponse,
    TrainingResultResponse,
    NovelListResponse
)
from model_tea.services import TrainingService
from model_tea.trainers.iterative import IterativeConfig
from model_tea.trainers.model import ModelConfig

router = APIRouter(prefix="/api/training", tags=["training"])
training_service = TrainingService()


@router.post("/model", response_model=TrainingResultResponse)
async def train_model(request: TrainModelRequest):
    """Train a model from model_mapping.json (primary method)"""
    config = ModelConfig()

    if request.max_iterations:
        config.max_iterations = request.max_iterations

    try:
        results = training_service.train_model(request.model_key, config)

        return TrainingResultResponse(
            success=True,
            message="Training completed successfully",
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/direct", response_model=TrainingResultResponse)
async def train_direct(request: TrainDirectRequest):
    """Train a novel directly from novels/ directory (dev/testing)"""
    config = IterativeConfig()

    if request.max_iterations:
        config.max_iterations = request.max_iterations
    if request.batch_size:
        config.batch_size = request.batch_size
    if request.learning_rate:
        config.learning_rate_start = request.learning_rate

    try:
        results = training_service.train_direct(request.novel_name, config)

        return TrainingResultResponse(
            success=True,
            message="Training completed successfully",
            training_time=results.get("training_time"),
            iterations=len(results.get("iterations", [])),
            final_quality=results.get("final_quality"),
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    status = training_service.get_training_status(job_id)
    return TrainingStatusResponse(**status)


@router.get("/novels", response_model=NovelListResponse)
async def list_novels():
    novels = training_service.list_available_novels()
    return NovelListResponse(novels=novels, total=len(novels))
