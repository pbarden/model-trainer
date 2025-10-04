from fastapi import APIRouter, HTTPException
from model_tea.api.schemas import ModelListResponse, ModelInfo
from model_tea.services import ChatService

router = APIRouter(prefix="/api/models", tags=["models"])
chat_service = ChatService()


@router.get("", response_model=ModelListResponse)
async def list_models(show_all: bool = False):
    models = chat_service.list_available_models(show_all=show_all)
    return ModelListResponse(models=models, total=len(models))


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    info = chat_service.get_model_info(model_name)

    if not info["exists"]:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    return ModelInfo(**info)
