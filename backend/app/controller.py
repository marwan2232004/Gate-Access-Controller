from sqlalchemy.orm import Session
from app.models import Plate

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

    # TODO: Call the actual plate detection function here
    # plate_number = detect_plate(image_data) 
    plate_number = "ABC1234" # Replace with the above line
    
    # FOR TESTING PURPOSES
    #----------------------------------------------
    #----------------------------------------------
    
    # save_dir = os.path.join(os.getcwd(), "uploaded_images")
    # os.makedirs(save_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # file_extension = image.filename.split(".")[-1]
    # saved_path = os.path.join(save_dir, f"{timestamp}_{plate_number}.{file_extension}")
    # print(saved_path)
    
    # with open(saved_path, "wb") as file:
    #     file.write(image_data)
    
    #----------------------------------------------
    #----------------------------------------------

    exists = plate_exists(db, plate_number)
    return {"allowed": exists, "plate": plate_number}

def get_all_plates(db: Session):
    plates = db.query(Plate).all()
    return [plate.plate_number for plate in plates]

def remove_all_plates(db: Session):
    db.query(Plate).delete()
    db.commit()
    return {"message": "All plates removed successfully"}