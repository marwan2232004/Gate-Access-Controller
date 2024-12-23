import os
import xml.etree.ElementTree as ET

import cv2
import imutils
import numpy as np


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

    if not load_annotation:
        return image

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
    if len(image.shape) == 2:
        return image

    height, width, channels = image.shape
    gray_image = image
    if channels == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif channels == 4:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        print("Unknown color system the image will remain unchanged")

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


def apply_morphological_operations(image, kernel_size=3):
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply morphological operations
    img_top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, structuring_element)
    img_black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, structuring_element)

    img_grayscale_plus_top_hat = cv2.add(image, img_top_hat)
    image = cv2.subtract(img_grayscale_plus_top_hat, img_black_hat)

    return image


def preprocess_image(image_path, image=None):
    # load image
    if image is None:
        image = cv2.imread(image_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        original_image = image

    # convert to gray scale
    gray_scale_image = convert_to_gray(original_image)
    # apply bilateral filter
    gray_filter = cv2.bilateralFilter(gray_scale_image, 13, 15, 15)

    gray_filter = apply_morphological_operations(gray_filter)

    binary_image_adaptive = cv2.adaptiveThreshold(
        gray_filter, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        9)

    return binary_image_adaptive


def find_contours(image):
    edged = np.uint8(image)  # Convert to an uint8 type

    # retrieves all the contours without establishing any hierarchical relationships.
    contours = cv2.findContours(edged.copy(),
                                cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    x_similarity_threshold = 10  # the threshold in x-axis to say that two contours have the same x coordinate
    y_similarity_threshold = 10  # the threshold in y-axis to say that two contours have the same x coordinate

    best_contours = []

    for contour in contours:
        # bounding rectangle returns the x,y of the top left corner and the width and height of the rect.
        x, y, w, h = cv2.boundingRect(contour)

        # Apply size and aspect ratio filters
        if 30 < w * h < 1500 and 4 > h / w > 1 and h > 10:
            is_duplicate = False

            # Check if the contour is duplicate
            for existing_x, existing_y, _, _ in best_contours:
                if (abs(x - existing_x) < x_similarity_threshold and
                        abs(y - existing_y) < y_similarity_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                best_contours.append([x, y, w, h])

    return best_contours


def get_y_coordinate(contour):
    _, y, _, _ = contour
    return y


def get_x_coordinate(contour):
    x, _, _, _ = contour
    return x


# Function to get the contours with common height as the plate numbers will be probably
# with the same height for the same car,
# so we can remove other contours which they will be not relevant
def get_common_contours_height(groups):
    tolerance = 20

    for i in range(len(groups)):
        group = groups[i]

        height = [h for _, _, w, h in group]

        # Determine the median area
        median_height = np.median(height)

        lower_bound = median_height - median_height * (tolerance / 100)
        upper_bound = median_height + median_height * (tolerance / 100)

        filtered_group = []
        for contour in group:
            height = contour[3]
            if lower_bound <= height <= upper_bound:
                filtered_group.append(contour)

        # Update the group with filtered contours
        groups[i] = filtered_group

    return groups


# Divide the contours to groups based on y-axis level
def divide_contours_to_groups_vertically(contours):
    # Sort the contours based on the y-coordinate
    sorted_contours = sorted(contours, key=get_y_coordinate)

    contours_group = [[]]
    group_idx = 0
    vertical_threshold = 15  # define the maximum vertical difference between two contours

    for i in range(len(sorted_contours)):
        if len(contours_group[group_idx]) == 0:
            contours_group[group_idx].append(sorted_contours[i])
            continue

        current_contour = sorted_contours[i]
        first_contour = contours_group[group_idx][0]

        current_y = current_contour[1]
        first_y = first_contour[1]

        vertical_check = current_y < first_y + vertical_threshold

        if vertical_check:
            contours_group[group_idx].append(sorted_contours[i])
        else:
            group_idx += 1
            contours_group.append([sorted_contours[i]])

    return contours_group


def get_horizontal_groups(sorted_horizontal_contours):
    groups = [[]]
    group_idx = 0
    overlap_threshold = 10
    horizontal_threshold = 30  # define the maximum horizontal difference between two contours
    for i in range(len(sorted_horizontal_contours)):

        if len(groups[group_idx]) == 0:
            groups[group_idx].append(sorted_horizontal_contours[i])
            continue

        current_x, _, current_w, _ = sorted_horizontal_contours[i]
        last_x, _, last_w, _ = groups[group_idx][-1]

        right_horizontal_check = (
                last_x + last_w + horizontal_threshold > current_x > last_x + last_w - overlap_threshold)

        if right_horizontal_check:
            groups[group_idx].append(sorted_horizontal_contours[i])
        else:
            group_idx += 1
            groups.append([sorted_horizontal_contours[i]])
    return groups


def divide_contours_to_groups_horizontally(contours_group):
    horizontal_groups = []
    for group in contours_group:
        if len(group) < 4:
            continue

        sorted_group = sorted(group, key=get_x_coordinate)
        horizontal_groups = horizontal_groups + get_horizontal_groups(sorted_group)

    return horizontal_groups


def find_plate_chars(image):
    # If you pass cv.CHAIN_APPROX_NONE,
    # all the boundary points are stored.
    # But actually do we need all the points?
    # For e.g., you found the contour of a straight line.
    # Do you need all the points on the line to represent that line?
    # No, we need just two end points of that line.
    # This is what cv.CHAIN_APPROX_SIMPLE does.
    # It removes all redundant points and compresses the contour,
    # thereby saving memory.
    best_groups = []  # list to store the best groups of contours that represent the characters of the plate

    best_contours = find_contours(image)

    # Divide the contours to groups based on y-axis level
    contours_group = divide_contours_to_groups_vertically(best_contours)

    # Divide the contours to groups based on x-axis level
    horizontal_groups = divide_contours_to_groups_horizontally(contours_group)

    horizontal_groups = get_common_contours_height(horizontal_groups)
    # ---------------------------------Grab the sequences lager than 4 contours--------------------------------
    for group in horizontal_groups:
        if len(group) < 4:
            continue
        best_groups.append(group)

    return best_groups


def get_chars_from_contours(contour, image):
    x, y, w, h = contour
    char = image[y:y + h, x:x + w]
    char = cv2.resize(char, (28, 28))
    char = apply_morphological_operations(char)
    binary_image_adaptive = cv2.adaptiveThreshold(
        char, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5)
    return binary_image_adaptive
