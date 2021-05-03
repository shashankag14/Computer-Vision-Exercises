import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
 
import histogram_module
import dist_module
 
def rgb2gray(rgb):
 
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
 
    return gray
 
#
# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image
#
def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):
 
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
 
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
 
    D = np.zeros((len(model_images), len(query_images)))
 
    best_match = np.zeros(len(model_images))
    for i in range(len(query_hists)):
        dummy_best_match = []
        for j in range(len(model_hists)):
            D[i][j] = dist_module.get_dist_by_name(query_hists[i], model_hists[j], dist_type)
            dummy_best_match.append(D[i][j])
 
        best_match[i] = np.argmin(dummy_best_match)
    return best_match, D
 
def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
 
    image_hist = []
 
    # compute hisgoram for each image and add it at the bottom of image_hist
    for i in image_list:
        img_color = np.array(Image.open(i))
        
        # Convert RGB to gray if needed
        if hist_isgray:
            img_gray = rgb2gray(img_color.astype('double'))            
            image_hist.append(histogram_module.get_hist_by_name(img_gray, num_bins, hist_type))
        
        else:
            image_hist.append(histogram_module.get_hist_by_name(img_color.astype('double'), num_bins, hist_type))
            
    return image_hist
 
#
# for each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# note: use the previously implemented function 'find_best_match'
# note: use subplot command to show all the images in the same Python figure, one row per query image
#
 
def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
 
    plt.figure()
    num_nearest = 5  # show the top-5 neighbors
 
    # Check if the histogram needed is to be gray/dxdy or rg/rgb
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
 
    # Compute histograms for all model and query images
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
 
    match_imgs = np.zeros((len(query_images), num_nearest))
 
    # Compute dsitnaces between each pair of histograms and save the smallest for each query image
    for i in range(len(query_hists)):
        dist_imgs = []
        for j in range(len(model_hists)):
            dist_imgs.append(dist_module.get_dist_by_name(query_hists[i],model_hists[j],dist_type))
 
        match_imgs[i] = np.argsort(dist_imgs)[:num_nearest]
   
    f, ax1 = plt.subplots(len(query_images), num_nearest + 1, figsize=(14,14))
 
    ax1[0][0].set_title('Query Image')
    ax1[0][1].set_title('Model Image 1')
    ax1[0][2].set_title('Model Image 2')
    ax1[0][3].set_title('Model Image 3')
    ax1[0][4].set_title('Model Image 4')
    ax1[0][5].set_title('Model Image 5')
 
 
    for j in range(len(query_images)):
        ax1[j][0].imshow(np.array(Image.open(query_images[j])))
        for k in range(num_nearest):
            im_name = match_imgs[j][k]
            ax1[j][k+1].imshow(np.array(Image.open(model_images[int(im_name)])))
 
    f.tight_layout()
    plt.show() 
