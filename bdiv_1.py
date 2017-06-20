import numpy as np


def bdiv_1(v):


    n = v.shape[3]
    m = v.shape[2]
    k = v.shape[0]
    
    div_v = np.zeros([k,m,n],dtype=np.complex64)
    
    
    div_v[:,:,0] = v[:,0,:,0]
    div_v[:,:,n-1] = -v[:,0,:,n-2]
    div_v[:,:,1:(n-1)] = v[:,0,:,1:(n-1)]-v[:,0,:,:(n-2)]
    
    div_v[:,0,:] = div_v[:,0,:] + v[:,1,0,:]
    div_v[:,m-1,:] = div_v[:,m-1,:] - v[:,1,m-1,:]
    div_v[:,1:m-1,:] = div_v[:,1:m-1,:] + (v[:,1,1:m-1,:] - v[:,1,:(m-2),:])

    return div_v
