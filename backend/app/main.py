from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import router as plate_router

app = FastAPI()
origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=False,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.include_router(plate_router, prefix="", tags=["plates"])
