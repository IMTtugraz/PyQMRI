import numpy as np

def fgrad_1(u):
    
    
#u = np.zeros([2,dimX,dimY]) #zum test

#    u = np.array(np.arange(1,16384*2+1))
#    u.shape = (2,128,128)

    n = u.shape[2]
    m = u.shape[1]
    k = u.shape[0]
    
    grad = np.zeros([k,2,m,n],dtype=np.complex64)
     
    grad[:,0,:,:(n-2)] = u[:,:,1:(n-1)] - u[:,:,:(n-2)]
    
    grad[:,1,:(m-2),:] = u[:,1:(m-1),:] - u[:,:(m-2),:]


    return grad

