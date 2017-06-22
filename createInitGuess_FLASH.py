import numpy as np
import scipy as sp



from TV import TV
#import scipy.sparse


def createInitGuess_FLASH(dat,FA,TR):
    

    
    
  imgs = abs(dat)
  #
  
  [numAlpha, dimSlice, dimX, dimY] = imgs.shape
  
  #    alpha = np.copy(FA)
  DESPOT1    = np.zeros([dimSlice,dimX,dimY],dtype='complex128')#np.complex128
  M0         = np.zeros([dimSlice,dimX,dimY],dtype='complex128')
  mask_guess = np.zeros([dimSlice,dimX,dimY],dtype='complex128')
  imgsize = dimX*dimY
  
  #    if numAlpha is not len(alpha):
  #        print('Error.Dimension missmatch')
  
  for iSlice in range(0,dimSlice):
  #        mask = np.ones([numAlpha, dimX,dimY])
      #MaskToUse = 1 wenn keine maskierung verwendet wird
  #        MaskToUse = 1 
  #        alphaCor = np.ones([numAlpha,dimX,dimY],dtype=np.complex64)
    X_row    = np.ones([numAlpha,dimY*dimX],dtype=np.complex64)
    Y_row    = np.ones([numAlpha,dimY*dimX],dtype=np.complex64)
    Y_img    = np.ones([numAlpha, dimX,dimY],dtype=np.complex64)
  
    for i in range (0,numAlpha):
  #            alphaCor[i,:,:] = alpha[i] * B1[iSlice,:,:]# * mask[MaskToUse, :,:]
  #            X = imgs[i,iSlice,:,:] * mask[MaskToUse,:,:] / np.tan(alphaCor[i,:,:])
  #            Y = imgs[i,iSlice,:,:] * mask[MaskToUse,:,:] / np.sin(alphaCor[i,:,:])               
      X = imgs[i,iSlice,:,:] / np.tan(FA[i,iSlice,:,:])
      Y = imgs[i,iSlice,:,:]  / np.sin(FA[i,iSlice,:,:])        
      X_row[i,:] = np.reshape(X, imgsize) #copy(X[:]).shape = (16384)
      Y_row[i,:] = np.reshape(Y, imgsize) #np.copy(Y)
      Y_img[i,:,:] = np.copy(Y)
  
    Y_eq = Y_row.T.flatten()#makescopy otherwise ravel()#p.reshape(Y_row.T, Y_row.T.shape[0] * Y_row.T.shape[1])  #ravel()
    X_eq = X_row.T.flatten()#np.reshape(X_row.T, 163840)
    X_ones = np.ones(X_eq.shape)
    
    #dimx = 2/numAlpha * len(X_eq) ?used
    dimy = len(X_eq)
    
    ind_row_X = np.arange(0,(len(X_eq)/numAlpha))  # 1 +1 ???
    ind_row_X = np.tile(ind_row_X,(numAlpha,1)) #(10, 16384)
    ind_row_X = ind_row_X.T.ravel()
    ind_row_X[np.isinf(ind_row_X)] = 0  #nocopy needed np.nan_to_num
    ind_row_X[np.isnan(ind_row_X)] = 0 

    dimyarr = np.arange(0,dimy)
    X_sparse = sp.sparse.coo_matrix((X_eq,(ind_row_X,dimyarr)),shape=(imgsize*2,imgsize*numAlpha)).tocsr()
    
    ind_row_ones =  np.array(np.arange((dimY*dimX),len(X_eq) / numAlpha+dimY*dimX)) #dtype?
    ind_row_ones = np.tile(ind_row_ones,(numAlpha,1))
    ind_row_ones = ind_row_ones.T.ravel()


    
    ones_sparse = sp.sparse.coo_matrix((X_ones,(ind_row_ones,dimyarr)),shape=(imgsize*2,imgsize*numAlpha)).tocsr()
#####oder lil_matrix    
    Sparse_Sys_Mat = X_sparse + ones_sparse
    Y_eq[np.isinf(Y_eq)] = 0  
    Y_eq[np.isnan(Y_eq)] = 0 
  
  #####t
  
  ####mit np.linalg.lstsq?
  
    Y_mean_tmp = np.mean(Y_img,0)
    #
    Y_mean = (np.tile(Y_mean_tmp.ravel(),(numAlpha,1))).T.flatten()
    Y_std_tmp = np.std(Y_img,0,ddof=1)
    
    Y_std = (np.tile(Y_std_tmp.ravel(),(numAlpha,1))).T.flatten()
    
    Y_eq = (Y_eq-Y_mean) / Y_std
    
    #Lamda 5e6
    sol_regtv = TV(Sparse_Sys_Mat, Y_eq, np.zeros([2,dimX,dimY]),1e6)
    sol_regtv = np.reshape(sol_regtv, [2,dimX,dimY])
    
    
    m_tv = sol_regtv[0,:,:] * Y_std_tmp
    #print(m_tv[m_tv < 0])
    #m_tv[isnan(m_tv)] = 1
    b_tv = sol_regtv[1,:,:] * Y_std_tmp + Y_mean_tmp
     # print(m_tv)
    DESPOT1[iSlice,:,:] = -TR / np.log(abs(m_tv))  #neg wird später berücksichtigt
    #print(np.isnan( DESPOT1[0,:,60:90]))
    M0[iSlice,:,:] = b_tv/(1-m_tv)
    
    #    mask_guess[iSlice,:,:] = mask[MaskToUse,:,:]
    DESPOT1 = np.squeeze(DESPOT1)
    print('Slice',iSlice+1,'of',dimSlice)
  
  return M0,DESPOT1,mask_guess

