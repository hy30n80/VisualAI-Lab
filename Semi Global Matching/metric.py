import numpy as np

def compute_mse(true_image, predicted_image):
    true_image = true_image.astype(np.float32)
    predicted_image = predicted_image.astype(np.float32)
    return np.mean((true_image - predicted_image) ** 2)

def compute_psnr(mse_value, max_pixel_value = 255.0):
    return 20 * np.log10(max_pixel_value) - 10 * np.log10(mse_value)