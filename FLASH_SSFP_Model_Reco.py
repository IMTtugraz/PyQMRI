#FLASH_SSFP_Model_Reco
import numpy as np
import time
import matplotlib.pyplot as plt


from nlinvns_maier import nlinvns
from ftimes import ftimes
from ifft2c import ifft2c
from compute_mask import compute_mask
from walsh_sens_2d import walsh_sens_2d
from createInitGuess_FLASH import createInitGuess_FLASH


class VFA_Model_Reco:
  def __init__(self):
    self.par = []
    self.data= []
    self.result = []
    self.guess = []
    self.images = []
    self.FT = []
    self.FTH = []

  def DESPOT1_Model_Reco(par):

    par.U = np.ones((par.y).shape, dtype=bool)
    par.U[abs(par.y) == 0] = False

  
    par.lowres = np.array(np.zeros([NScan,NSlice,par.dimX,par.dimY]), dtype = np.complex64)

    for i in range(NScan):
        par.lowres[i,:,:] = np.sum(np.fft.ifft2(par.y[i,:,:,:,:])*np.conj(par.C[None,:,None,:,:]),axis = 1)
    #Generate low res data:
#    cal_shape = (100,100)#par.dimY, par.dimX) # in future; block-size = nPH/acc
#
#    f = np.hamming(cal_shape[0])[:,None].dot(np.hamming(cal_shape[1])[:,None].T)
#    fmask = np.zeros([par.dimX,par.dimY])
#    row_gap = int((par.dimY - cal_shape[1])/2)  #par.dimY = len(fmask[0,:]) //X[:,0]
#    col_gap = int((par.dimX - cal_shape[0])/2)  #if fmask defined different
#    fmask[row_gap:row_gap+f.shape[1],col_gap:col_gap+f.shape[0]] = f
#
#    fmask = np.tile(fmask[None, ...], [par.NC,1,1])
#
#    fmask = np.ones([par.NC,par.dimY,par.dimX]) #???
#
#    #cal_im = np.array()   ##added
#    
#    print('deriving M(TI(1)) and coil profiles')
#    for a in range(0,par.NScan):
#        for i in range(0,par.NSlice):
#            combinedData = (np.squeeze(np.sum(par.y[a,:,i,:,:][None,...],0)))
#
#            filtered_cal_data = np.squeeze(combinedData * fmask) #check richtig
#            
#            cal_im = fftshift2(ifft2c(filtered_cal_data))  # checck richtig
#            [recon,csm_walsh] = walsh_sens_2d(cal_im) 
#
#            par.lowres[a,i,:,:] = recon # 10,1,128,128
##
###    
#    del combinedData,nlinvout, sumSqrC   

####################################
##B1 correction
#not needed for now

    par.fa_corr = np.ones([par.NSlice,par.dimX,par.dimY])
# =============================================
## initial guess for the parameter maps
# =============================================
### create an initial guess by standard pixel-based fitting of "composite images"
    
    th = time.clock()

    [M0_guess, T1_guess, mask_guess] = createInitGuess_FLASH(par.lowres[0:par.NScan_FLASH,:,:,:],par.fa,par.TR,par.fa_corr)

    T1_guess[np.isnan(T1_guess)] = np.spacing(1)
    T1_guess[np.isinf(T1_guess)] = np.spacing(1)
    T1_guess[T1_guess<0] = 0 
    T1_guess[T1_guess>5000] = 5000
    T1_guess = np.abs(T1_guess)

    M0_guess[M0_guess<0] = 0 
    M0_guess[np.isnan(M0_guess)] = np.spacing(1)
    M0_guess[np.isinf(M0_guess)] = np.spacing(1)   
    

    hist =  np.histogram(np.abs(M0_guess),int(1e2))
    aa = np.array(hist[0], dtype=np.float64)
    #bb = hist[1] #hist0[1][:-1] + np.diff(hist0[1])/2
    bb = np.array(hist[1][:-1] + np.diff(hist[1])/2, dtype=np.float64)
   
    idx = np.array(aa > 0.01*aa[0],dtype=np.float64)

    M0_guess[M0_guess > bb[int(np.sum(idx))]] = bb[int(np.sum(idx))] #passst
    #print(M0_guess)
    M0_guess = np.squeeze(M0_guess)

    mask_guess = compute_mask(M0_guess,False)

    par.mask = mask_guess#par.mask[:,63] is different
    
    par.T1_sc = np.max(T1_guess)
    par.M0_sc = np.max(np.abs(M0_guess))
    
    #print(mask_guess)
    print('T1 scale: ',par.T1_sc,
                              '/ M0_guess: ',par.M0_sc)
    #print(M0_guess[39,11]) M0 guess is gleich

    M0_guess = M0_guess / par.M0_sc
    T1_guess = T1_guess / par.T1_sc

    T1_guess[np.isnan(T1_guess)] = 0;
    M0_guess[np.isnan(M0_guess)] = 0;
    
    par.T1_guess = T1_guess * par.T1_sc
    par.M0_guess = M0_guess * par.M0_sc
#        
    print( 'done in', time.clock() - th)

    result = np.array([T1_guess*par.T1_sc,M0_guess*par.M0_sc])
    guess = result

    return result,guess, par