#nlinvns
#% Written and invented
#% by Martin Uecker <muecker@gwdg.de> in 2008-09-22
#%
#% Modifications by Tilman Sumpf 2012 <tsumpf@gwdg.de>:
#%	- removed fftshift during reconstruction (ns = "no shift")
#%	- added switch to return coil profiles
#%	- added switch to force the image estimate to be real
#%	- use of vectorized operation rather than "for" loops
#%
#% Version 0.1
#%
#% Biomedizinische NMR Forschungs GmbH am
#% Max-Planck-Institut fuer biophysikalische Chemie
import numpy as np
import time
from helper_fun.fftshift2 import fftshift2
#import pyfftw
import pyopencl as cl
import pyopencl.array as clarray
from gridroutines import gridding

DTYPE = np.complex64
DTYPE_real = np.float32

def NUFFT(N,NScan,NC,NSlice,traj=None,dcf=None,trafo=0,mask=None):
  platforms = cl.get_platforms()
  ctx = cl.Context(
            dev_type=cl.device_type.GPU,
            properties=[(cl.context_properties.PLATFORM, platforms[1])])
  queue=[]
  queue.append(cl.CommandQueue(ctx,platforms[1].get_devices()[0]))
  FFT = gridding(ctx,queue,4,2,N,NScan,(NScan*NC*NSlice,N,N),(1,2),traj,dcf,N,1000,DTYPE,DTYPE_real,radial=trafo,mask=mask)
  return (ctx,queue[0],FFT)


def nlinvns(Y, n,*arg):  #*returnProfiles,**realConstr):

    (NC,Nproj,N) = Y.shape

    nrarg = len(arg)
    if nrarg == 2:
        returnProfiles = arg[0]
        realConstr     = arg[1]
    elif nrarg < 2:
        realConstr = False
        if nrarg < 1:
            returnProfiles = 0

    print(Y.strides)
    (ctx,queue,fft) = NUFFT(N,1,NC,1,mask=np.ones_like(Y))




    print('Start...')

    alpha = 1e0

    [c, y, x] = Y.shape

    if returnProfiles:
        R = np.zeros([c+2, n, 1, y, x],DTYPE)

    else:
        R = np.zeros([2, n, 1, y,x],DTYPE)

    img_shape = (1,c,1,y,x)
    ksp_shape = (1,c,1,y,x)

#    test_data = (np.random.randn(ksp_shape[0],ksp_shape[1],ksp_shape[2],ksp_shape[3],ksp_shape[4])+1j*np.random.randn(ksp_shape[0],ksp_shape[1],ksp_shape[2],ksp_shape[3],ksp_shape[4])).astype(DTYPE)
#    test_img = (np.random.randn(ksp_shape[0],ksp_shape[1],ksp_shape[2],ksp_shape[3],ksp_shape[4])+1j*np.random.randn(ksp_shape[0],ksp_shape[1],ksp_shape[2],ksp_shape[3],ksp_shape[4])).astype(DTYPE)
#
#
#    data_test = clarray.to_device(fft.queue[0],np.require((test_data.astype(DTYPE)),requirements='C'))
#    img_test =  clarray.to_device(fft.queue[0],np.require((test_img.astype(DTYPE)),requirements='C'))
#    cl_out1 = clarray.zeros(fft.queue[0],img_shape,dtype=DTYPE)
#    cl_out2 = clarray.zeros(fft.queue[0],ksp_shape,dtype=DTYPE)
#
#    fft.adj_NUFFT(cl_out2,img_test)
#    fft.fwd_NUFFT(cl_out1,data_test)
#
#    cl_out1 = cl_out1.get()
#    cl_out2 = cl_out2.get()
#
#    a = np.vdot(cl_out1.flatten(),test_img.flatten())
#    b = np.vdot(test_data.flatten(),cl_out2.flatten())
#    test = np.abs(a-b)
#    print("test deriv-op-adjointness streamed:\n <xx,DGHyy>=%05f %05fi\n <DGxx,yy>=%05f %05fi  \n adj: %.2E"  % (a.real,a.imag,b.real,b.imag,(test)/(x**2)))





    # initialization x-vector
    X0 = np.array(np.zeros([c+1, 1,y, x]),dtype=DTYPE)  #5,128,128
    X0[0,:,:] = 1	#object part


    # initialize mask and weights
#    P = np.ones(Y[0,:,:].shape,dtype=DTYPE)  #128,128
#    P[Y[0,:,:] == 0] = 0


    W = weights(x, y) #W128,128

#    P = fftshift2(P)  #128,128
    W = fftshift2(W).astype(DTYPE)[None,None,None,...]
    #Y = fftshift2(Y)  # 4,128,128

    #normalize data vector
    yscale = 100 / np.sqrt(scal(Y,Y))
    YS = (Y[None,:,None,...] * yscale).astype(DTYPE) ##check
    #YS = np.round(YS,4) #4,128,128


    XT = np.zeros([c+1,1,y,x],dtype=DTYPE)   #5,128,128
    XN = np.copy(X0)   #5,128,128

    start = time.clock()
    for i in range(0,n):

        # the application of the weights matrix to XN
        # is moved out of the operator and the derivative
        XT[0,...] = np.copy(XN[0,...])
        XT[1:,...] = apweightsns(W, np.copy(XN[1:,...]),fft,img_shape)  #W((+1)128,128)[None,...] (5,128,128)

        RES = (YS - opns(XT,fft,ksp_shape))


        print(np.round(np.linalg.norm(RES)))#check
#        print(RES.shape)  4,128,128


        #calculate rhs
        r = derHns(W,XT,RES,realConstr,fft,img_shape,ksp_shape) ##128,128  128,128   5,128,128  4,128,128

        #r.shape = (5,128,128)

        r = np.array(r + alpha * (X0 - XN),dtype=DTYPE)


        z = np.zeros_like(r)
        d = np.copy(r);
        dnew = np.linalg.norm(r)**2
        dnot = np.copy(dnew)

#        if np.any(~np.isfinite(r)):
#          print("test")
#        if np.any(~np.isfinite(d)):
#          print("test")
#        if np.any(~np.isfinite(dnew)):
#          print("test")
#        if np.any(~np.isfinite(dnot)):
#          print("test")
        for j in range(0,500):

            #regularized normal equations
            q = derHns( W, XT, derns(W,XT,d,fft,ksp_shape,img_shape), realConstr,fft,img_shape,ksp_shape) + alpha * d
            q[~np.isfinite(q)] = 0

            a = dnew/ np.real(scal(d,q))
            z = z + a*(d)
            r = r - a * q

            dold = np.copy(dnew)
            dnew = np.linalg.norm(r)**2

            d = d*((dnew / dold)) + r

#            print("Sqrt Dnew: %f, 1e-2 * dnot: %f"%(np.sqrt(dnew),1e-2*dnot))
            if (np.sqrt(dnew) < (1e-2 * dnot)):
                break


        print('(',j,')')

        XN = XN + z

        alpha = alpha / 3

        #postprocessing

        CR = apweightsns(W, XN[1:,...],fft,img_shape)

        if returnProfiles:
            R[2:,i,...] = CR / yscale #,6,9,128,128

#        print(CR.shape)
        C = (np.conj(CR) * CR).sum((0,1))

#        print(XN.shape)
#        print(C.shape)
        R[0,i,...] =  (XN[0,...] * np.sqrt(C) / yscale)
        R[1,i,...] = np.copy(XN[0,...])

    R = (R)
    end = time.clock()  #sec.process time
    print('done in', round((end - start)),'s')
    del fft, ctx, queue
    return np.squeeze(R)



def scal(a,b):#check
    v = np.vdot(a,b)#np.array(np.sum(np.conj(a) * b),dtype=np.complex64)
    return v

def apweightsns(W,CT,fft,img_shape):
    tmp = clarray.to_device(fft.queue[0],(W * CT).astype(DTYPE))
    C = clarray.zeros(fft.queue[0],img_shape,DTYPE)

    fft.adj_NUFFT(C,tmp)
    return C.get()

def apweightsnsH(W,CT,fft,ksp_shape):
    tmp = clarray.to_device(fft.queue[0],CT.astype(DTYPE))
    C = clarray.zeros(fft.queue[0],ksp_shape,DTYPE)

    fft.fwd_NUFFT(C,tmp)
    C = np.conj(W) * C.get()
    return C

def opns(X,fft,ksp_dim):
    K = X[0,...]*X[1,...]
    tmp2 = clarray.to_device(fft.queue[0],K)
    tmp1 = clarray.zeros(fft.queue[0],ksp_dim,DTYPE)
    fft.fwd_NUFFT(tmp1,tmp2)
#    K = np.array(P*tmp2.get(),dtype=np.complex64)  #[None,...]
    #K = np.round(K,4)
    return tmp1.get()

def derns(W,X0,DX,fft,ksp_dim,img_dim):
    K = X0[0,...] * apweightsns(W,DX[1:,...],fft,img_dim)
#    print(K.shape)
#    print(DX[0,...].shape)
#    print(X0[0,...].shape)
    K = K + (DX[0,...] * X0[1:,...])   #A# 2/1

    tmp2 = clarray.to_device(fft.queue[0],K)
    tmp1 = clarray.zeros(fft.queue[0],ksp_dim,DTYPE)
    fft.fwd_NUFFT(tmp1,tmp2)
#    K = P * K.get()
    return tmp1.get()


def derHns(W,X0,DK,realConstr,fft,img_dim,ksp_dim):

    K = clarray.zeros(fft.queue[0],img_dim,DTYPE)
    tmp = clarray.to_device(fft.queue[0],DK)
    fft.adj_NUFFT(K,tmp)
    K = K.get()


    if realConstr:
        DXrho = np.sum(np.real(K * np.conj(X0[1:,...])),(0,1) )
    else:
        DXrho = np.sum( K * np.conj(X0[1:,...]),(0,1) )[None,...]


    DXc = apweightsnsH(W, (K * np.conj(X0[0,...])),fft,ksp_dim)


    DX = np.squeeze(np.concatenate((DXrho[None,...],DXc),axis = 1))[:,None,...]
    return DX

#def nsFft(M):
#    si = M.shape
#    a = 1 / (np.sqrt((si[M.ndim-1])) * np.sqrt((si[M.ndim-2])))
#    K = np.array((pyfftw.interfaces.numpy_fft.fft2(M,norm=None)).dot(a),dtype=np.complex64)#
#    return K
#
#
#def nsIfft(M):
#    si = M.shape
#    a = np.sqrt(si[M.ndim-1]) * np.sqrt(si[M.ndim-2])
#    #K = np.array(np.fft.ifftn(M, axes=(0,)),dtype=np.float64) #*a
#    K = np.array(pyfftw.interfaces.numpy_fft.ifft2(M,norm=None).dot(a))
#    return K #.T


def weights(x,y):
    W = np.zeros([x,y])
    for i in range(0,x):
        for j in range(0,y):
            d = ((i) / x - 0.5)**2 + ((j) / y - 0.5)**2
            W[j,i] = 1 / (1 + 220 * d)**16 ###16
    return W










