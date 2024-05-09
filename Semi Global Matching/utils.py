import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def disparity_map_show(image):
    # disparity map 시각화
    plt.imshow(image, cmap='gray')  # colormap을 'jet'으로 설정하여 시각적으로 더 잘 나타내도록 함
    plt.colorbar()  # 컬러바 추가
    plt.title('Disparity Map')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()