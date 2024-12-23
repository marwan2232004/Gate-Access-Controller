from sqlalchemy.orm import Session
from app.models import Plate
import sys, os
import numpy as np
import cv2
sys.path.append(os.path.abspath('../'))
from inference.inference import get_car_plate_characters
from utils.utils import show_images
sys.path.append(os.path.abspath('../backend'))

def plate_exists(db: Session, plate_number: str) -> bool:
    return db.query(Plate).filter(Plate.plate_number == plate_number).first() is not None

def add_plate(db: Session, plate_number: str):
    if plate_exists(db, plate_number):
        return {"message": "Plate already exists"}
    
    plate = Plate(plate_number=plate_number)
    db.add(plate)
    db.commit()
    return {"message": "Plate added successfully"}

def remove_plate(db: Session, plate_number: str):
    if plate_exists(db, plate_number):
        db.delete(plate_number)
        db.commit()
        return {"message": "Plate removed successfully"}
    return {"message": "Plate not found"}

def is_allowed(db: Session, image):
    image_data = image.file.read()

    image_array = np.frombuffer(image_data, np.uint8)
    image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    plate_number = get_car_plate_characters(image_path=None, image=image)

    exists = plate_exists(db, plate_number)
    return {"allowed": exists, "plate": plate_number}

def get_all_plates(db: Session):
    plates = db.query(Plate).all()
    return [plate.plate_number for plate in plates]

def remove_all_plates(db: Session):
    db.query(Plate).delete()
    db.commit()
    return {"message": "All plates removed successfully"}