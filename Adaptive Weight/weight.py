import numpy as np


def adaptive_weight(window, gamma_c=5, gamma_p=17.5):
    window_size = window.shape[0]

    window_center = (window_size - 1) // 2  #

    center_rgb = window[window_center, window_center, :]

    color_distance_array = np.zeros((window_size, window_size), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            rgb = window[i, j]
            color_distance = np.sqrt(np.sum((center_rgb - rgb) ** 2))
            color_distance_array[i, j] = color_distance
    distance_weight = np.zeros((window_size, window_size))


    for i in range(window_size):
        for j in range(window_size):
            distance = np.sqrt((i - window_center) ** 2 + (j - window_center) ** 2)
            distance_weight[i, j] = distance
    weight_mask = np.exp(-(color_distance_array/gamma_c + distance_weight/gamma_p))

    return weight_mask