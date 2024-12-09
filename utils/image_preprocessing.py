import os
import xml.etree.ElementTree as ET

import cv2


# Function to load an image and its corresponding annotation
def load_image_and_annotation(image_name,
                              image_dir,
                              annotation_dir,
                              load_annotation=False):
    # 1- get image path
    # 2- read image
    # 3- convert to RGB for display
    # When using cv2.imread(), the image is loaded in BGR order, not RGB.
    # so we need to convert it to RGB to display it correctly
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not load_annotation: return image

    # get the file extension from the image path
    extension = image_path.split('.')[1]

    print(f'file extension: {extension}')

    # Load corresponding XML annotation
    annotation_path = os.path.join(annotation_dir, image_name.replace(f'.{extension}', '.xml'))
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Get bounding box coordinates from the XML for the plate location
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Draw the bounding box on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return image


def convert_to_gray(image):
    height, width, channels = image.shape
    gray_image = image
    if channels == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif channels == 4:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        print("Unkown color system the image will remain unchanged")

    return gray_image


# The input image must be gray
# cv2.INTER_AREA is best for downscaling the image
def resize_with_aspect_ratio(gray_image,
                             width=None,
                             height=None,
                             keep_aspect_ratio=False,
                             interpolation=cv2.INTER_AREA):
    # Get the original width and height
    original_height, original_width = gray_image.shape
    # Set the original value to original dimensions

    if width is None and height is None: return gray_image

    if width is None: width = original_width
    if height is None: height = original_height

    if not keep_aspect_ratio: return cv2.resize(gray_image, (width, height), interpolation=interpolation)

    if width is not None:
        scaling_factor = width / original_width
    else:
        scaling_factor = height / original_height

    new_size = (int(original_width * scaling_factor), int(original_height * scaling_factor))

    resized_image = cv2.resize(gray_image, new_size, interpolation=interpolation)

    return resized_image


# Apply adaptive thresholding
# cv2.adaptiveThreshold(): This function applies adaptive thresholding where the threshold value
# is calculated based on the local neighborhood of each pixel.
# It is particularly useful when the image has different lighting across different regions.

# 255 (Max Value): This is the maximum value that will be assigned to pixels
# that exceed the dynamically calculated threshold in each region.

# cv2.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the local neighborhood minus a constant C.
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is the weighted sum of the local neighborhood minus a constant C.

# block size of a local region
# must be odd
def convert_to_binary(image, block_size=31, constant=2):
    binary_image_adaptive = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
    )
    return binary_image_adaptive


def preprocess_image(image_name, image_dir, annotation_dir,
                     resized_width=300, resized_height=200,
                     keep_aspect_ratio=False,
                     resize_interpolation=cv2.INTER_AREA,
                     adaptive_threshold_block_size=31,
                     adaptive_threshold_constant=2,
                     load_annotation=False):
    # load image
    original_image = load_image_and_annotation(image_name,
                                               image_dir,
                                               annotation_dir,
                                               load_annotation)
    # convert to gray scale
    gray_scale_image = convert_to_gray(original_image)
    # resize image
    resized_image = resize_with_aspect_ratio(gray_scale_image,
                                             width=resized_width,
                                             height=resized_height,
                                             keep_aspect_ratio=keep_aspect_ratio,
                                             interpolation=resize_interpolation)
    # convert to binary using adaptive thresholding
    binary_image_adaptive = convert_to_binary(resized_image,
                                              block_size=adaptive_threshold_block_size,
                                              constant=adaptive_threshold_constant)
    return binary_image_adaptive
