import numpy as np
from numpy import histogram as hist
import sys
sys.path.append("../filter-Q1")
from gauss_module import gaussderiv

#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    # flat_img : Flatening the greyscale img for easy computation
    flat_img = img_gray.flatten()
   
    # max_intervals : Number of intervals for num_bins
    max_intervals = 255/num_bins 
    hists = np.zeros(num_bins, np.int32)
    
    # Provides the indices at which value of histogram will be incremented
    bin_idx = flat_img / max_intervals
    for idx in bin_idx:
        hists[int(idx)] += 1
   
    # Normalizing the histogram values
    hists = hists/hists.sum()
    
    # Stores bins at regular intervals of bin_gap
    # One extra interval has been added to make the size of bins consistent with the code in identification.py (plt.bar)
    bins = np.arange(0, max_intervals * (num_bins+1) , max_intervals)
   
    return hists, bins


#  compute joint histogram for each color channel in the image, histogram should be normalized so that sum of all values equals 1
#  assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
def rgb_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    # Range of RGB - [0,255]
    max_intervals = 255/num_bins

    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            # ...
            R = img_color[i,j,0]/max_intervals
            G = img_color[i,j,1]/max_intervals
            B = img_color[i,j,2]/max_intervals

            # Truncate the indices upto num_bins
            R_idx = min(R,num_bins)
            G_idx = min(G,num_bins)
            B_idx = min(B,num_bins)
            hists[int(R_idx),int(G_idx),int(B_idx)] += 1

    # normalize the histogram such that its integral (sum) is equal 1
    hists = hists/hists.sum()
    hists = hists.reshape(hists.size)

    return hists

#  compute joint histogram for r/g values
#  note that r/g values should be in the range [0, 1];
#  histogram should be normalized so that sum of all values equals 1
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
def rg_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    # Range of normalised R,G - [0,1]
    max_intervals = 1.0/num_bins   
    
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            R = (img_color[i,j,0])
            G = (img_color[i,j,1])
            B = (img_color[i,j,2])
            
            # Normalising R,G pixel wise
            RGB_sum = R + G + B
            R_norm, G_norm = R / RGB_sum, G / RGB_sum

            # Truncate the indices upto num_bins-1
            R_idx = min(R_norm / max_intervals, num_bins-1)
            G_idx = min(G_norm / max_intervals, num_bins-1)

            hists[int(R_idx), int(G_idx)] += 1

    hists = hists/hists.sum()
    hists = hists.reshape(hists.size)    
    return hists

#  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
#  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
#  histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input grayvalue image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  note: you can use the function gaussderiv.m from the filter exercise.
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    
    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    # compute the first derivatives
    img_dx,img_dy = gaussderiv(img_gray,7.0)
    
    # quantize derivatives to "num_bins" number of values
    max_intervals = 60/num_bins  
    
    # Since indices cant be negative, shifting img_dx,img_dy values from [-30,30]-->[0,60]
    img_dx_q = (img_dx + 30) / max_intervals
    img_dy_q = (img_dy + 30) / max_intervals
   
    for i in range(img_dx_q.shape[0]):
        for j in range(img_dx_q.shape[1]):
            hists[ int(img_dx_q[i,j]), int(img_dy_q[i,j]) ] += 1 
    
    # ...
    hists = hists/hists.sum()
    hists = hists.reshape(hists.size)
    return hists

def is_grayvalue_hist(hist_name):
    if hist_name == 'grayvalue' or hist_name == 'dxdy':
        return True
    elif hist_name == 'rgb' or hist_name == 'rg':
        return False
    else:
        assert False, 'unknown histogram type'


def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
    if dist_name == 'grayvalue':
        return normalized_hist(img1_gray, num_bins_gray)
    elif dist_name == 'rgb':
        return rgb_hist(img1_gray, num_bins_gray)
    elif dist_name == 'rg':
        return rg_hist(img1_gray, num_bins_gray)
    elif dist_name == 'dxdy':
        return dxdy_hist(img1_gray, num_bins_gray)
    else:
        assert 'unknown distance: %s'%dist_name

