import numpy as np
import math

# 
# compute chi2 distance between x and y
#
def dist_chi2(x,y):
    # your code here
    eps = 1e-10
    for (a, b) in zip(x, y) :
        if (a == 0):
            chi_dis = np.sum((a - b) ** 2)
        else :
            chi_dis = np.sum(((a - b) ** 2) / (a))
    #chi_dis = np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(x, y)])
    return chi_dis

# 
# compute l2 distance between x and y
#
def dist_l2(x,y):
    # your code here
    l2_dis = math.sqrt(np.sum([((a - b) ** 2) for (a, b) in zip(x, y)]))
    return l2_dis

# 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
    # your code here
    intersect_dis = 1 - np.sum([min(a,b) for (a, b) in zip(x, y)])  # --- TODO 256 or n
    return intersect_dis
 
def get_dist_by_name(x, y, dist_name):
    if dist_name == 'chi2':
        return dist_chi2(x,y)
    elif dist_name == 'intersect':
        return dist_intersect(x,y)
    elif dist_name == 'l2':
        return dist_l2(x,y)
    else:
        assert'unknown distance: %s'%dist_name





