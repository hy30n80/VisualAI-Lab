import numpy as np
from tqdm import tqdm

def compute_aggregated_cost(h, w, d, cost_volume_by_r, dh, dw):
    p1 = 5
    p2 = 150
    new_h = h-dh
    new_w = w-dw
    if (0 <= new_h < height) and (0 <= new_w < width):
        prev_with_same_disparity = cost_volume_by_r[new_h, new_w, d]
        prev_with_upper_disparity = cost_volume_by_r[new_h, new_w, d+1] + p1 if d + 1 < disparity else np.inf
        prev_with_lower_disparity = cost_volume_by_r[new_h, new_w, d-1] + p1 if d - 1 >= 0 else np.inf
        prev_with_lowest = np.min(cost_volume_by_r[new_h, new_w, :])
        update_value = min(prev_with_same_disparity, prev_with_lower_disparity, prev_with_upper_disparity, prev_with_lowest+p2) - prev_with_lowest
    
    else:
        update_value = 0

    return update_value


# output : aggregated_costs (height, width, disparity)
def cost_aggregation_SGM(cost_volume):

    global height
    global width 
    global disparity 

    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparity = cost_volume.shape[2]

    original_cost_volume = cost_volume 
    aggregated_cost_volume = np.zeros((height, width, disparity))

    
    #directions = [(1,0), (0,1), (-1,0), (0,-1)]
    directions = [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1) ]
    #directions = [(0, 1), (0, 2), (1, 1), (2, 2), (1, 0), (2, 0), (1, -1), (2, -2), (0, -1), (0, -2), (-1, -1), (-2, -2), (-1, 0), (-2, 0), (-1, 1), (-2, 2)]



    progress_bar = tqdm(total = len(directions) * height * width * disparity)

    #(0,0) 출발
    for (dh, dw) in directions[:4]:
        cost_volume_by_r = np.zeros((height, width, disparity))
        for h in range(0, height):
            for w in range(0, width):
                for d in range(0, disparity):
                    cost_volume_by_r[h,w,d] = original_cost_volume[h,w,d] + compute_aggregated_cost(h,w,d, cost_volume_by_r, dh, dw)   
                    aggregated_cost_volume[h,w,d] += cost_volume_by_r[h,w,d]
                    progress_bar.update(1)


    #(height, width) 출발
    for (dh, dw) in directions[4:]:
        cost_volume_by_r = np.zeros((height, width, disparity))
        for h in range(height-1, 1, -1):
            for w in range(width-1, 1, -1):
                for d in range(0, disparity):
                    cost_volume_by_r[h,w,d] = original_cost_volume[h,w,d] + compute_aggregated_cost(h,w,d, cost_volume_by_r, dh, dw)   
                    aggregated_cost_volume[h,w,d] += cost_volume_by_r[h,w,d]
                    progress_bar.update(1)
  

    progress_bar.close()
    return aggregated_cost_volume    

