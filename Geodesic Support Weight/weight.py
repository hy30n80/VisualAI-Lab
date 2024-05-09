import numpy as np
import math


def compute_geodesic_support_weight(window, window_size = 31, gamma=10):

    #input : (31,31,3) window
    #output : (31,31) weight_mask

    w = window_size 
    c = window_size // 2 + 1
    weight_mask = np.full((w+2,w+2), 1e5) #Initialize with a large value 
    weight_mask[c,c] = 0 
    window = np.pad(window, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0) #padded window

    for step in range(1,window_size):
        tuples = [(dx, dy) for dx in range(-c+1,c) for dy in range(-c+1,c) if abs(dx) + abs(dy) == step]
        center = [c,c]

        for (dx,dy) in tuples:
            x = center[0] + dx
            y = center[1] + dy
            
            right = weight_mask[x+1,y] + math.sqrt(np.sum((window[x,y] - window[x+1,y])**2))
            left = weight_mask[x-1,y] + math.sqrt(np.sum((window[x,y] - window[x-1,y])**2))
            up = weight_mask[x,y-1] + math.sqrt(np.sum((window[x,y] - window[x,y-1])**2))
            down = weight_mask[x,y+1] + math.sqrt(np.sum((window[x,y] - window[x,y+1])**2))

            weight_mask[x,y] = np.exp(-min(right, left, up, down)/gamma)
            #weight_mask[x,y] = gamma/min(right, left, up, down)

    #Excluding padding region
    weight_mask = weight_mask[1:w+1, 1:w+1]
    #print(weight_mask)
    return weight_mask
