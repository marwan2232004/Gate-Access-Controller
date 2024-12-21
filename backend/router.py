from fastapi import APIRouter, Form, Depends
from sqlalchemy.orm import Session
import controller
from database import get_db

router = APIRouter()

@router.post("/add-plate")
async def add_plate_route(plate_number: str = Form(...), db: Session = Depends(get_db)):
    return controller.add_plate(db, plate_number)

@router.delete("/remove-plate/{plate_number}")
async def remove_plate_route(plate_number: str, db: Session = Depends(get_db)):
    return controller.remove_plate(db, plate_number)

@router.get("/is-allowed")
async def detect_plate_route(plate_number: str, db: Session = Depends(get_db)):
    return controller.is_allowed(db, plate_number)

@router.get("/get-all")
async def get_all_plates_route(db: Session = Depends(get_db)):
    return controller.get_all_plates(db)

@router.delete("/remove-all")
async def remove_all_plates_route(db: Session = Depends(get_db)):
    return controller.remove_all_plates(db)
