"""
FastAPI Application Assembly Point
"""

from fastapi import FastAPI

from api.config import settings
from api.routers import health, predict

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
)

# Include Modular Routers
app.include_router(health.router)
app.include_router(predict.router)
