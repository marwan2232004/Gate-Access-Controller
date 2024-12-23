import cv2
import joblib
import numpy as np
import pytesseract
from ultralytics import YOLO

from utils.image_preprocessing import preprocess_image, find_plate_chars, get_chars_from_contours


# Now you can resize each image in the dataset
def extract_features(dataset_images):
    features = []
    for img in dataset_images:
        if img is None:
            print("Warning: Skipping None image")
            continue
        img_resized = cv2.resize(img, (28, 28))
        img_flattened = img_resized.flatten()
        features.append(img_flattened)
    return features


def get_car_plate_characters(image_path):
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path=image_path)

    # Step 2: Detect the license plate characters
    best_groups = find_plate_chars(preprocessed_image)

    # Step 3: crops the characters from the image
    chars = []
    for group in best_groups:
        for contour in group:
            char = get_chars_from_contours(contour, preprocessed_image)
            chars.append(char)

    # Step 4: Load the model
    model = joblib.load("models/OCR_SVM.pkl")

    # Step 5: Predict the characters
    predictions = ""
    for char in chars:
        features = np.array(extract_features([char])).reshape(-1, 1).flatten()
        predictions += model.predict([features])[0]

    return predictions


def get_car_plate_characters_yolo(image_path):
    best = YOLO(model="models/yolo_fine_tuned.pt")
    results = best.predict(conf=0.25, source=image_path)
    image = cv2.imread(image_path)
    predictions = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers

            # Crop the image using the bounding box
            cropped_img = image[y1:y2, x1:x2]

            # Ensure the cropped image is C-contiguous
            cropped_img = np.ascontiguousarray(cropped_img)

            # Save the cropped image temporarily
            temp_path = f"./car_plate"
            cv2.imwrite(temp_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            text = pytesseract.image_to_string(cropped_img, config='--psm 7')
            predictions.append(text.strip())

    return predictions
