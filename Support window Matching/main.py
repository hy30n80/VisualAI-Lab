import numpy as np
from stereo import compute_disparity_map_WWTA, compute_disparity_score, disparity_map_show
from utils import read_colour_image, read_grayscale_image, write_image
from datetime import datetime


if __name__ == "__main__":
    image_left = read_colour_image('Image/scene1.row3.col3.ppm')
    image_right = read_colour_image('Image/scene1.row3.col5.ppm')
    dmap_gt = read_grayscale_image("Image/truedisp.row3.col3.pgm")

    
    #hyperparameter # 7, 15, 31 
    window_size = 7

    #Window based WTA
    dmap = compute_disparity_map_WWTA(image_left, image_right, window_size = window_size)
    rmse = compute_disparity_score(dmap_gt, dmap)
    
    print(f"Disparity Score (RMS): {rmse:.2f}")

    disparity_map_show(dmap)

    write_image("result")

