import numpy as np

from skimage import filters
from skimage.morphology import remove_small_objects #binaryopening for numpy
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_opening
from scipy.ndimage import binary_closing
from scipy.ndimage import iterate_structure
from scipy.ndimage import generic_filter
from scipy.ndimage.morphology import binary_fill_holes

def compute_mask (mask,is_irgn):

    mask = abs(mask)
    mask[np.isnan(mask)] = 0
    mask[np.isinf(mask)] = 0

    mask = (mask - np.min(mask)) /  (np.max(mask)-np.min(mask))
    

    Sim = mask # TODO:

    #thresh = mask[np.round(x/2),np.round(y*3/4)]     ## better with cv
    thresh = filters.threshold_otsu(mask)
    print('thresh')
    print(thresh)
    #thresh = 0.35490196078
    #plt.imshow(S)
    if is_irgn:
        BW1 =  np.array(Sim/1.5 > thresh)
    else:
        BW1 =  np.array(Sim*1.2 > thresh)  
    #plt.imshow(Sim)
    
    #a = np.zeros((5, 5))
    #a[2, 2] = 1
    #print(BW1)
    circle2 = np.array(generate_binary_structure(2, 1))
    #cc = np.array(binary_dilation(a, structure=circle3).astype(a.dtype))
    #circle3 = np.array(iterate_structure(circle2 , 2),dtype=int)
    circle4 = np.array(iterate_structure(circle2 , 10))
    
    #bwperim
    def test_func(values):
        return values.sum()     
    res = generic_filter(BW1,test_func,footprint=circle2,mode='constant')
    BW2 = np.copy(BW1)
    BW2[res==5] = 0
    #

    mask_body = remove_small_objects(BW2, min_size=100, connectivity=8)
    #
    
    if is_irgn:
        BW1 = np.array(binary_opening(BW1,circle2),dtype = int)
        BW_final = remove_small_objects(BW1, min_size=(np.sum(BW1)/3), connectivity=8)
        BW_final = binary_fill_holes(BW_final).astype('int') #scipy.ndimage.morphology.binary_fill_holes
        BW_final = np.array(binary_closing(BW_final,circle4),dtype = int)
    else:
        BW3 = binary_fill_holes(mask_body).astype('int')#BW3 = #fill-holes
        BW_final = remove_small_objects(BW3, min_size=100, connectivity=8)
    #    
    mask_new = np.array(BW_final)   
    #
    return mask_new