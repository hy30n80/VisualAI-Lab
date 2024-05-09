import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from weight import compute_geodesic_support_weight


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


def compute_disparity_map_WTA(Il, Ir, window_size = 31, gamma=10):
    h = 288
    w = 384
    r = window_size // 2
    max_disparity = 32
    Id = np.zeros((h,w), dtype=int)
    total_iterations = len(range(r, h-r)) * len(range(r, w-r))
    progress_bar = tqdm(total = total_iterations)

    for y in range(r, h-r):
        for x in range(r, w-r):
            window_left = Il[y-r:y+r+1, x-r:x+r+1, :]
            window_right = Ir[y-r:y+r+1, x-r:x+r+1, :]

            weight_mask_left = adaptive_weight(window_left)

            minimum_SAD = 1e8 

            disparity = 0
            for d in range(1,max_disparity):
                if x-r-d < 0:
                    break
                
                window_right = Ir[y-r:y+r+1, x-r-d:x+r-d+1, :]  # 동일한 상이 오른쪽 이미지에 왼쪽에 존재하기 때문에 X축 (-) 방향으로 탐색 
                wieght_mask_right = adaptive_weight(window_right)
                
                # (w,w,1)
                weight = np.multiply(weight_mask_left, wieght_mask_right)
                current_SAD = np.sum(np.multiply(weight, np.sum(np.abs(window_left - window_right), axis=-1))) / np.sum(weight) #SAD
                
                
                if current_SAD < minimum_SAD:
                    minimum_SAD = current_SAD
                    disparity = d
                
            Id[y, x] = disparity 
            progress_bar.update(1)

    progress_bar.close()
    return Id




def compute_disparity_score(It, Id): 
    #It 이미지에서 0이 아닌 값의 위치를 나타내는 마스크 (유효한 disparity 값만 확인하고자 하는 의미)

    mask = It != 0
    N = np.sum(mask)

    if N != 0:
        rms = np.sqrt(np.sum((Id[mask] - It[mask]) ** 2) / N)

    return rms


def disparity_map_show(image):
    # disparity map 시각화
    plt.imshow(image, cmap='gray')  # colormap을 'jet'으로 설정하여 시각적으로 더 잘 나타내도록 함
    plt.colorbar()  # 컬러바 추가
    plt.title('Disparity Map')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()