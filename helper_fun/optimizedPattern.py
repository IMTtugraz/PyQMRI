import numpy as np
#import matplotlib.pyplot as plt

def optimizedPattern(data, AF,ACL):
    
    [dimY,dimX,NScan,NCoil,NAlpha] = data.T.shape   #NScan wieder weggelassen
    
    P = np.zeros(data.shape,dtype = np.complex64)
    
    # Radial Pattern
    NSpokes = np.floor(dimY/AF);
    shift = np.pi / (NSpokes * NAlpha);
    spoke_shift = np.arange(0,shift*NAlpha-shift/2, shift)
    
    def pol2cart(r,phi):
        return r * np.exp(1j*phi)
    
    for k in range(0,NAlpha):
        r = (dimX/2)**2+(dimY/2)**2
        r = np.arange(-np.sqrt(r),np.sqrt(r)-1)
        theta = np.linspace(spoke_shift[k],np.pi+spoke_shift[k],np.floor(np.pi/2*dimY/AF))
        [thg,rg] = np.meshgrid(theta,r)
        xn = rg * np.cos(theta) #pol->cart
        yn = rg * np.sin(theta) #
        test = np.round(xn)+np.abs(np.round(np.min(xn))) #-1...used as ind.
        test2 = np.round(yn)+np.abs(np.round(np.min(yn)))#
        
        mask = np.zeros([test.shape[0]-1,test.shape[0]-1],dtype = np.complex64)#size noch festlegen
        nspokes = test.T.shape[0]
        print(np.max(test))
        for i in range(0,nspokes):
            for j in range(0,test.shape[0]):#test>180???
                if test[j,i] < 180 and test2[j,i] < 180:
                    mask[test2[j,i],test[j,i]] = 1
         
        
        [yMask, xMask] = mask.shape
        xCut = xMask - dimX
        yCut = yMask - dimY
    
        mask = mask[int(np.floor(yCut/2)):int(np.floor(yMask-yCut/2)),int(np.floor(xCut/2)):int(np.floor(xMask-xCut/2))]
        mask = np.tile(mask, [NCoil,NScan,1,1])
        P[k,:,:,:,:] = data[k,:,:,:,:] * mask
    
            
