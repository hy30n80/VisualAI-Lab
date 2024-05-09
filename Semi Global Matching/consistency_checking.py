import numpy as np
import cv2

def stereo_matching(img_left, img_right, block_size=5, max_disparity=64):
    # Perform block matching
    stereo = cv2.StereoBM_create(numDisparities=max_disparity, blockSize=block_size)
    disparity_left = stereo.compute(img_left, img_right)

    # Perform left-right consistency check
    stereo_right = cv2.StereoBM_create(numDisparities=max_disparity, blockSize=block_size)
    disparity_right = stereo_right.compute(img_right, img_left)

    # Initialize disparity map
    height, width = img_left.shape[:2]
    disparity_map = np.zeros((height, width), dtype=np.float32)

    # Threshold for left-right consistency check
    threshold = 1

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            d = int(disparity_left[y, x])

            # Check if disparity is valid
            if d < max_disparity:
                # Calculate corresponding pixel in right image
                x_right = x - d

                # Perform left-right consistency check
                if 0 <= x_right < width:
                    if np.abs(int(disparity_right[y, x_right]) - d) < threshold:
                        disparity_map[y, x] = d

    return disparity_map