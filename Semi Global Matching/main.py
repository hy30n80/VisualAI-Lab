import numpy
from utils import read_colour_image, read_grayscale_image, write_image, disparity_map_show
from datetime import datetime
from cost_calculation import cost_calculation_AE
from cost_aggregation import cost_aggregation_SGM
from disparity_computation import compute_disparity
from metric import compute_mse, compute_psnr
import cv2

if __name__ == "__main__":
    left_image = read_colour_image('Image/scene1.row3.col3.ppm')
    right_image = read_colour_image('Image/scene1.row3.col5.ppm')
    ground_truth = read_grayscale_image("Image/truedisp.row3.col3.pgm")

    original_cost_volume = cost_calculation_AE(ref_image = left_image, tar_image = right_image, disaprity = 32, ref="left")
    #original_cost_volume = cost_calculation_AE(ref_image = right_image, tar_image = left_image, disaprity = 32, ref="right")

    aggregated_cost_volume = cost_aggregation_SGM(original_cost_volume)
    disparity_map = compute_disparity(aggregated_cost_volume)
    mse = compute_mse(true_image = ground_truth, predicted_image=disparity_map)
    print("Mean Squared Error: ",mse)
    disparity_map_show(disparity_map)
    cv2.imwrite('./left_dmap.png', disparity_map)

