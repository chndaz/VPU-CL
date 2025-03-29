import os
import cv2
import numpy as np


def normalize_brightness(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    normalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return normalized_image


def adaptive_brightness_enhancement(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def process_images(input_folder, output_folder, enhancement_type="normalize"):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # if filename.lower().endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {filename}")
            continue

        if enhancement_type == "adaptive":
            processed_image = adaptive_brightness_enhancement(image)
        else:
            processed_image = normalize_brightness(image)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed_image)
        print(f"complete : {filename} ({enhancement_type})")


if __name__ == "__main__":
    input_path = r"D:\python_new\statistical_modeling\data\an"
    output_path = r"D:\python_new\statistical_modeling\data\up_an"
    enhancement_type = "adaptive"  # choose "normalize" 或 "adaptive"
    process_images(input_path, output_path, enhancement_type)