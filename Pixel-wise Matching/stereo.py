import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def compute_disparity_map_PWTA(Il, Ir):
    h = 288
    w = 384
    max_disparity = 32

    Id = np.zeros((h,w), dtype=int)
    total_iterations = h*w
    progress_bar = tqdm(total = total_iterations)

    for y in range(0, h):
        for x in range(0, w):
            
            window_left = Il[y, x, :]
            window_right = Ir[y, x, :]
            minimum_SAD = np.sum(np.abs(window_left - window_right))

            disparity = 0
            for d in range(1,max_disparity):
                if x-d < 0:
                    break
                
                # 동일한 상이 오른쪽 이미지에 왼쪽에 존재하기 때문에 X축 (-) 방향으로 탐색 
                window_right = Ir[y, x-d, :]
                current_SAD = np.sum(np.abs(window_left - window_right))

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