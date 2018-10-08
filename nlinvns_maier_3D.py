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
from fftshift3 import fftshift3
import pyopencl as cl
import pyopencl.array as clarray
from gridroutines import gridding

DTYPE = np.complex64
DTYPE_real = np.float32

def NUFFT(N,NScan,NC,NSlice,traj,dcf,trafo=1):
  platforms = cl.get_platforms()
  ctx = cl.Context(
            dev_type=cl.device_type.GPU,
            properties=[(cl.context_properties.PLATFORM, platforms[1])])
  queue=[]
  queue.append(cl.CommandQueue(ctx,platforms[1].get_devices()[0]))
  if trafo:
    FFT = gridding(ctx,queue,4,2,N,NScan,(NScan*NC,NSlice,N,N),(1,2,3),traj.astype(DTYPE),np.require(np.abs(dcf),DTYPE_real,requirements='C'),N,1000,DTYPE,DTYPE_real,radial=trafo)
  else:
    FFT = gridding(ctx,queue,4,2,N,NScan,(NScan*NC,NSlice,N,N),(1,2,3),traj,dcf,N,1000,DTYPE,DTYPE_real,radial=trafo)
  return (ctx,queue[0],FFT)




def nlinvns(Y, n, traj,dcf, trafo=1, *arg):  #*returnProfiles,**realConstr):

    (NC,NSlice,Nproj,N) = Y.shape
    nrarg = len(arg)
    if nrarg == 2:
        returnProfiles = arg[0]
        realConstr     = arg[1]
    elif nrarg < 2:
        realConstr = False
        if nrarg < 1:
            returnProfiles = 0

    (ctx,queue,fft) = NUFFT(N,1,NC,NSlice,traj,dcf,trafo)
    print('Start...')

    alpha = 1e-2

    ksp_shape = Y[None,...].shape

    [c, sl] = Y.shape[0:2]
    y = int(N/2)
    x = int(N/2)

    img_shape = (1,c,sl,y,x)

    if returnProfiles:
        R = np.zeros([c+2, n, sl, y, x],DTYPE)

    else:
        R = np.zeros([2, n, sl, y,x],DTYPE)


    # initialization x-vector
    X0 = np.array(np.zeros([1, c, sl, Nproj, N]),np.complex64)  #5,128,128
    image_0 = np.array(np.ones([sl, y, x]),np.complex64)	#object part


    # initialize mask and weights
#    P = np.ones(Y[0,...].shape,dtype=np.float64)  #128,128
#    P[Y[0,...] == 0] = 0


    W = weights(sl,Nproj,N) #W128,128

#    P = fftshift3(P)  #128,128
#    W = fftshift3(W)
    #Y = fftshift2(Y)  # 4,128,128

    #normalize data vector
    yscale = 100 / np.sqrt(scal(Y,Y))
    YS = Y * yscale ##check
    #YS = np.round(YS,4) #4,128,128


#    XT = np.zeros([c+1,z,y,x],dtype=np.complex64)   #5,128,128
    image_T = np.zeros([sl,y,x],dtype=np.complex64)
    image_N = np.ones([sl,y,x],dtype=np.complex64)
    XT = np.zeros([c,sl,y,x],dtype=np.complex64)
    XN = np.copy(X0)   #5,128,128

    start = time.clock()
    for i in range(0,n):

        # the application of the weights matrix to XN
        # is moved out of the operator and the derivative
        image_T = np.copy(image_N)
        XT = apweightsns(W, XN,fft,img_shape)  #W((+1)128,128)[None,...] (5,128,128)

        RES = (YS - opns(image_T,XT,fft,ksp_shape))


        print(np.round(np.linalg.norm(RES)))#check
#        print(RES.shape)  4,128,128


        #calculate rhs
        (r_img,r) = derHns(W,image_T,XT,RES,realConstr,fft,img_shape,ksp_shape) ##128,128  128,128   5,128,128  4,128,128

        #r.shape = (5,128,128)

        r = np.array(r + alpha * (X0 - XN),dtype=np.complex64)
        r_img = r_img + alpha * (image_0-image_N)


        z = np.zeros_like(r)
        z_img = np.zeros_like(r_img)
        d = np.copy(r)
        d_img = np.copy(r_img)

        dnew = np.linalg.norm(r)**2+np.linalg.norm(r_img)**2
        dnot = np.copy(dnew)

        for j in range(0,128):

            #regularized normal equations
#            q = derHns(P, W, XT, derns(W,XT,d,fft), realConstr,fft) + alpha * d

            (tmpim, tmpc) = derHns(W,image_T, XT, derns(W, image_T, XT, d_img,d,fft,ksp_shape,img_shape), realConstr,fft,img_shape,ksp_shape)
            q_img =  tmpim  + alpha * d_img
            q   =  tmpc   + alpha * d
            q = np.nan_to_num(q)
            q_img = np.nan_to_num(q_img)

            a = dnew / (np.real(scal(d,q))+np.real(scal(d_img,q_img)))

#            a = dnew/ np.real(scal(d,q))

            z = z + a * d
            z_img = z_img + a*d_img
            r = r - a * q
            r_img = r_img - a*q_img
            r = np.nan_to_num(r)
            r_img = np.nan_to_num(r_img)

            dold = np.copy(dnew)
            dnew = np.linalg.norm(r)**2+np.linalg.norm(r_img)**2

            d = d*((dnew / dold)) + r
            d_img = d_img * (dnew/dold) + r_img
            d = np.nan_to_num(d)
            d_img = np.nan_to_num(d_img)
            if (np.sqrt(dnew) < (1e-2 * dnot)):
                break


        print('(',j,')')

        XN = XN + z
        image_N = image_N + z_img

        alpha = alpha / 3

        #postprocessing

        CR = apweightsns(W, XN,fft,img_shape)

        if returnProfiles:
            R[2:,i,...] = CR / yscale #,6,9,128,128

        C = (np.conj(CR) * CR).sum((0,1))

        R[0,i,...] =  (image_N * np.sqrt(C) / yscale)
        R[1,i,...] = np.copy(image_N)

    R = (R)
    end = time.clock()  #sec.process time
    print('done in', round((end - start)),'s')
    return R



def scal(a,b):#check
    v = np.array(np.sum(np.conj(a) * b),dtype=np.complex64)
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

def opns(img,X,fft,ksp_dim):
    K = np.array(img*X,dtype=np.complex64)
    tmp2 = clarray.to_device(fft.queue[0],K)
    tmp1 = clarray.zeros(fft.queue[0],ksp_dim,DTYPE)
    fft.fwd_NUFFT(tmp1,tmp2)
#    K = np.array(P*tmp2.get(),dtype=np.complex64)  #[None,...]
    #K = np.round(K,4)
    return tmp1.get()

def derns(W,img,X0,DIMG,DX,fft,ksp_dim,img_dim):
    K = img * apweightsns(W,DX,fft,img_dim)
    K = K + (DIMG * X0)   #A# 2/1

    tmp2 = clarray.to_device(fft.queue[0],K)
    tmp1 = clarray.zeros(fft.queue[0],ksp_dim,DTYPE)
    fft.fwd_NUFFT(tmp1,tmp2)
#    K = P * K.get()
    return tmp1.get()


def derHns(W,img,X0,DK,realConstr,fft,img_dim,ksp_dim):


    K = clarray.zeros(fft.queue[0],img_dim,DTYPE)
    tmp = clarray.to_device(fft.queue[0],DK)
    fft.adj_NUFFT(K,tmp)
    K = K.get()

    if realConstr:
        DXrho = np.sum(np.real(K * np.conj(X0)),(0,1) )
    else:
        DXrho = np.sum( K * np.conj(X0),(0,1) )


    DXc = apweightsnsH(W, (K * np.conj(img)),fft,ksp_dim)
    return (DXrho,DXc)


def weights(z,y,x):
  W = np.zeros([z,y,x])
  for k in range(z):
    for j in range(y):
      for i in range(x):
          d = ((i) / x - 0.5)**2 #+  ((k) / z - 0.5)**2
          W[k,j,i] = 1 / (1 + 220 * d)**16
  return W










