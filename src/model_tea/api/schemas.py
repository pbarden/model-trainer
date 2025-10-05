from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class TrainModelRequest(BaseModel):
    model_key: str
    max_iterations: Optional[int] = None


class TrainDirectRequest(BaseModel):
    novel_name: str
    max_iterations: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str


class ChatGenerateRequest(BaseModel):
    model_name: str
    prompt: str
    max_length: Optional[int] = Field(default=250)  # Uses settings.default_max_length
    temperature: Optional[float] = Field(default=0.5)  # Uses settings.default_temperature
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 30
    repetition_penalty: Optional[float] = 1.4


class ChatGenerateResponse(BaseModel):
    response: str
    model_name: str
    prompt: str


class ModelInfo(BaseModel):
    name: str
    path: str
    exists: bool
    size_mb: Optional[float] = None
    model_type: Optional[str] = None
    vocab_size: Optional[int] = None


class ModelListResponse(BaseModel):
    models: List[str]
    total: int


class NovelListResponse(BaseModel):
    novels: List[str]
    total: int


class TrainingResultResponse(BaseModel):
    success: bool
    message: str
    training_time: Optional[float] = None
    iterations: Optional[int] = None
    final_quality: Optional[float] = None
    results: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    status: str
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
