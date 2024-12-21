from sqlalchemy.orm import Session
from models import Plate


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
    plate = db.query(Plate).filter(Plate.plate_number == plate_number).first()
    if plate:
        db.delete(plate)
        db.commit()
        return {"message": "Plate removed successfully"}
    return {"message": "Plate not found"}

def is_allowed(db: Session, plate_number: str):
    return {"is_allowed": plate_exists(db, plate_number)}

def get_all_plates(db: Session):
    plates = db.query(Plate).all()
    return [plate.plate_number for plate in plates]

def remove_all_plates(db: Session):
    db.query(Plate).delete()
    db.commit()
    return {"message": "All plates removed successfully"}