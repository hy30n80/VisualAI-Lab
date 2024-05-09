import cv2
import numpy as np

def read_colour_image(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(int)
    return image


def read_grayscale_image(filepath):
    image = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
    image = image.astype(int)
    return image


def write_image(filepath, image):
    adjusted_image = np.clip(image, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)
    cv2.imwrite(filepath, adjusted_image)
