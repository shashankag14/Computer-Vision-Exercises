import numpy as np
import math

def dist_chi2(x,y):
    dist_chi2_val = []
    for (i, j) in zip(x, y):
        # Corner case : i=0 leads to division by zero.
        if (i != 0):
            dist_chi2_val.append(((i - j)**2) / i)
    return sum(dist_chi2_val)

# compute l2 distance between x and y
def dist_l2(x,y):
    dist_l2_val = math.sqrt(sum((x-y)**2))
    return dist_l2_val
 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similar histograms
def dist_intersect(x,y):
    dist_intersect_val = 1-sum([min(x[i],y[i]) for i in range(x.size)])
    return dist_intersect_val

def get_dist_by_name(x, y, dist_name):
    if dist_name == 'chi2':
        return dist_chi2(x,y)
    elif dist_name == 'intersect':
        return dist_intersect(x,y)
    elif dist_name == 'l2':
        return dist_l2(x,y)
    else:
        assert 'unknown distance: %s'%dist_name