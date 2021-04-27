# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
# To access convolve2d function
from scipy import signal

def gauss(sigma):
    # Create a vector from [-3sigma, 3sigma]
    x = np.arange(start = -3 * sigma, stop = 3 * sigma, step = 1)
    
    # 1-D Gaussian values for the above x values
    Gx = []
    for i in x:
        val = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp( -i**2 / (2 * sigma**2))
        Gx = np.append(Gx, val)
        
    return Gx, x

def gaussderiv(img, sigma):
    # Using gauss and gaussdx functions to get the filter and derivative of filter
    [dG, val] = gaussdx(sigma)
    [G, val] = gauss(sigma)
    
    # Derivative of filter along y axis
    DY = np.zeros((len(val), len(val)))
    DY = dG[:,np.newaxis].dot(G[np.newaxis,:]) 

    # Derivative of filter along x axis
    DX = np.zeros((len(val), len(val)))
    DX = G[:,np.newaxis].dot(dG[np.newaxis,:]) 
    
    # Convolution of img with DX and DY
    imgDx = signal.convolve2d(img, DX)
    imgDy = signal.convolve2d(img, DY)
    
    return imgDx, imgDy

def gaussdx(sigma):
    # Create a vector from [-3sigma, 3sigma]
    x = np.arange(start = -3 * sigma, stop = 3 * sigma, step = 1)
    D = []
    for i in x:
        val = - (1 / (math.sqrt(2 * math.pi) * sigma**3)) * i * math.exp( -i**2 / (2 * sigma**2))
        D = np.append(D, val)
    return D, x

def gaussianfilter(img, sigma):
    [Gx,x] = gauss(sigma)
    
    gauss_2d = np.zeros((len(x), len(x)))
    gauss_2d = Gx[:,np.newaxis].dot(Gx[np.newaxis,:])
    
    # Convolution of img with 2d filter
    outimage = signal.convolve2d(img, gauss_2d)
    return outimage