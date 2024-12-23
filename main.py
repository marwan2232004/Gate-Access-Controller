import os

import cv2

from inference.inference import get_car_plate_characters, get_car_plate_characters_yolo
from utils.utils import show_images

IMAGES_DIR = os.path.join("dataset", "images")
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 200

if __name__ == "__main__":
    sample_image_name = "Cars96.png"
    image_path = os.path.join(IMAGES_DIR, sample_image_name)

    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictions = get_car_plate_characters(image_path=image_path)
    # display the image
    show_images([original_image], [f"Car Plate Number: {predictions}"])

    yolo_predications = get_car_plate_characters_yolo(image_path=image_path)
    # display the image
    show_images([original_image], [f"Car Plate Number: {yolo_predications}"])
