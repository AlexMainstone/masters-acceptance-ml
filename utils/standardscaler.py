import numpy as np

# Returns difference in mean and deviation of data
def scale(data):
    # iterate through features
    datat = data.T
    for i in range(len(datat)):
        datat[i] = (datat[i] - np.mean(datat[i])) / np.std(datat[i])
    return datat.T