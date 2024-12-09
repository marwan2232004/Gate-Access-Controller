import os

from utils.image_preprocessing import preprocess_image

IMAGES_DIR = os.path.join("data", "car-plate-detection", "images")
ANNOTATIONS_DIR = os.path.join("data", "car-plate-detection", "annotations")
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 200


def main(image_name):
    # Step 1: Preprocess the image
    preprocessed_img = preprocess_image(image_name=image_name,
                                        image_dir=IMAGES_DIR,
                                        annotation_dir=ANNOTATIONS_DIR,
                                        adaptive_threshold_block_size=31,
                                        resized_width=IMAGE_WIDTH,
                                        resized_height=IMAGE_HEIGHT)

    # Step 2: Detect the license plate

    # Step 3: Segment characters

    # Step 4: Recognize characters

    # Step 5: Access Control


if __name__ == "__main__":
    sample_image_name = "Cars0.png"
    main(sample_image_name)
