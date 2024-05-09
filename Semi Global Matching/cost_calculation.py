import numpy as np

# output : cost volume (height, width, disparity)
def cost_calculation_AE(ref_image, tar_image, disaprity = 32, ref="left"):

    height = ref_image.shape[0]
    width = ref_image.shape[1]
    disparity = disaprity
    cost_volume = np.zeros((height, width, disparity))

    if ref == "left":
        for h in range(0, height):
            for w in range(0, width):
                for d in range(0, disparity):
                    if w-d >= 0:
                        cost_volume[h,w,d] = np.sum(np.abs(ref_image[h,w] - tar_image[h,w-d]))
                    else: 
                        cost_volume[h,w,d] = np.inf

    elif ref == "right":
        for h in range(0, height):
            for w in range(0, width): 

                for d in range(0, disparity):
                    if w+d < width:
                        cost_volume[h,w,d] = np.sum(np.abs(ref_image[h,w] - tar_image[h,w+d]))
                    else: 
                        cost_volume[h,w,d] = np.inf


    return cost_volume