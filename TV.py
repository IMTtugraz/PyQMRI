import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from fgrad_1 import fgrad_1
from bdiv_1 import bdiv_1

def TV(A,d,x0,lambdaa):#lambda fkt....
#    print('d')
#    print(d)
    #Tau = Stepsize, theta is used to weight x_bar
    
    tau = 1/np.sqrt(8)
    sigma = 1/np.sqrt(8)
    theta = 1
    gamma = 0.9 * lambdaa
    
    maxiter = 200
    
    [alpha,m,n] = x0.shape  #np.zeros([2,dimX,dimY])
    x_bar = np.copy(x0)

    x = np.copy(x0)


    y = np.zeros([2,x0.shape[0],x0.shape[1],x0.shape[2]],dtype=np.complex64)
    
    Ata = A.dot(A.T)
    At = A.T  #copy?
    Ata[Ata == np.inf] = np.spacing(1)  #äquiv. for eps, wrong!!!!
    Ata[Ata == np.nan] = np.spacing(1)
    At[At == np.inf] = np.spacing(1)
    At[At == np.inf] = np.spacing(1)
   
    go_on = True; 
    count = 1;


    while go_on:
#        print('x1')
#        print(xbar)
        grad_xbar = fgrad_1(x_bar); 
        y_hat = y+(sigma*(grad_xbar))
#        print('x2')
#        print(xbar)
        y = y_hat / np.maximum(1,np.sqrt(sum(y**2,1)))
        
        #primal update
        div_y = bdiv_1(y)

        x_hat = x-(tau*div_y)

        #(I + tau dG)^(-1)
        

        B = identity(n*m*alpha)+tau*lambdaa*(Ata)

        v = (tau*lambdaa*A.dot(d)) + (x_hat.ravel())

        v[np.isnan(v)] =  np.spacing(1)  ##isnan geht net
        x_new = spsolve(B,v)   #B*x = v

        #extrapolation-step
        x_bar = x_new + theta * (x_new-x.flatten())  #32768
        #print(s)
        x_bar = np.reshape(x_bar, x0.shape) #tictoc mit np.reshape()
        x = np.copy(x_new)
        #xbar = np.copy(x_bar)
        #unsinn von zeile 77
        
        if count >=  maxiter:
            go_on = False;
        

        x.shape = x0.shape
        count += 1

    x = np.reshape(x, [alpha,m,n])    #[2,dimX,dimY]
    return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    