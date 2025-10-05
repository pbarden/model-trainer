from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from model_tea.api.routes import training_router, chat_router, models_router
from model_tea.config.settings import settings

app = FastAPI(
    title="Model Tea API",
    description="API for Model Tea training and inference system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training_router)
app.include_router(chat_router)
app.include_router(models_router)


@app.get("/")
async def root():
    return {
        "name": "Model Tea API",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


def main():
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
