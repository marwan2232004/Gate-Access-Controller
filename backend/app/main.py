from fastapi import FastAPI
from app.router import router as plate_router

app = FastAPI()

app.include_router(plate_router, prefix="", tags=["plates"])
