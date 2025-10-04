from fastapi import APIRouter, HTTPException
from model_tea.api.schemas import ChatGenerateRequest, ChatGenerateResponse
from model_tea.services import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])
chat_service = ChatService()


@router.post("/generate", response_model=ChatGenerateResponse)
async def generate_text(request: ChatGenerateRequest):
    if not chat_service.load_model(request.model_name):
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_name}")

    try:
        response = chat_service.generate(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )

        return ChatGenerateResponse(
            response=response,
            model_name=request.model_name,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
