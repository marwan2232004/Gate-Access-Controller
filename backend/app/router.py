from fastapi import APIRouter, Form, Depends
from sqlalchemy.orm import Session
import app.controller
from app.database import get_db
from fastapi import UploadFile, File

router = APIRouter()

@router.post("/add-plate")
async def add_plate_route(plate_number: str = Form(...), db: Session = Depends(get_db)):
    return app.controller.add_plate(db, plate_number)

@router.delete("/remove-plate/{plate_number}")
async def remove_plate_route(plate_number: str, db: Session = Depends(get_db)):
    return app.controller.remove_plate(db, plate_number)

@router.post("/is-allowed")
async def detect_plate_route(image: UploadFile = File(...), db: Session = Depends(get_db)):
    return app.controller.is_allowed(db, image)

@router.get("/get-all")
async def get_all_plates_route(db: Session = Depends(get_db)):
    return app.controller.get_all_plates(db)

@router.delete("/remove-all")
async def remove_all_plates_route(db: Session = Depends(get_db)):
    return app.controller.remove_all_plates(db)
