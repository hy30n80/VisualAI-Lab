def compute_disparity(aggregated_cost_volume):
    disparity_map = aggregated_cost_volume.argmin(axis=2)
    return disparity_map
