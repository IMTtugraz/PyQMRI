import numpy as np

import pyopencl as cl
import pyopencl.array as clarray
import reikna.cluda as cluda
from reikna.fft import  FFTShift
from gpyfft.fft import FFT

from helper_fun.calckbkernel import calckbkernel

class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class gridding:
  def __init__(self, ctx,queue, par,kwidth=4,overgridfactor=2,fft_dim=(1,2),klength=10000,DTYPE=np.complex64,DTYPE_real=np.float32, radial=True, SMS=False):
    self.radial = radial
    self.DTYPE = DTYPE
    self.DTYPE_real = DTYPE_real
    self.fft_shape = (par["NScan"]*par["NC"]*(par["par_slices"]+par["overlap"]),par["N"],par["N"])
    self.traj = par["traj"]
    self.dcf = par["dcf"]
    self.ctx = ctx
    self.queue = queue
    if self.radial:
      (self.kerneltable,self.kerneltable_FT,self.u) = calckbkernel(kwidth,overgridfactor,par["N"],klength)
      self.kernelpoints = self.kerneltable.size
      self.api = cluda.ocl_api()
      self.fft_scale = DTYPE_real(self.fft_shape[-1])
      self.deapo = 1/self.kerneltable_FT.astype(DTYPE_real)
      self.kwidth = kwidth/2
      self.cl_kerneltable = cl.Buffer(self.queue[0].context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.kerneltable.astype(DTYPE_real).data)
      self.deapo_cl = cl.Buffer(self.queue[0].context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.deapo.data)
      self.dcf = clarray.to_device(self.queue[0],self.dcf)
      self.traj = clarray.to_device(self.queue[0],self.traj)
      self.tmp_fft_array = (clarray.zeros(self.queue[0],self.fft_shape,dtype=DTYPE))
      self.fft2 = []
      self.fftshift = []
      self.thr = []
      for j in range(len(queue)):
        self.thr.append(self.api.Thread(queue[j]))
        fftshift = FFTShift(self.tmp_fft_array,axes=fft_dim)
        self.fftshift.append(fftshift.compile(self.thr[j]))
        fft = FFT(ctx,queue[j],self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],out_array=self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],axes=fft_dim)
        self.fft2.append(fft)
      self.gridsize = par["N"]
      self.fwd_NUFFT = self.NUFFT
      self.adj_NUFFT = self.NUFFTH
    else:
      if SMS:
        self.packs = par["packs"]
        self.shift = clarray.to_device(self.queue[0],par["shift"].astype(np.int32))
        self.fwd_NUFFT = self.FFT_SMS
        self.adj_NUFFT = self.FFTH_SMS
        self.fft_shape = (int(self.fft_shape[0]/par["packs"]),self.fft_shape[1],self.fft_shape[2])
      else:
        self.fwd_NUFFT = self.FFT
        self.adj_NUFFT = self.FFTH
      self.fft_scale = DTYPE_real(np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
      self.tmp_fft_array = (clarray.zeros(self.queue[0],self.fft_shape,dtype=DTYPE))
      self.fft2 = []
      self.fft_shape = self.fft_shape
      self.par_fft = int(self.fft_shape[0]/par["NScan"])
      self.mask = clarray.to_device(self.queue[0],par["mask"])
      for j in range(len(queue)):
        fft = FFT(ctx,queue[j],self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],out_array=self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],axes=fft_dim)
        self.fft2.append(fft)
    if DTYPE==np.complex128:
      print('Using double precission')
      self.prg = Program(ctx, open('./Kernels/OpenCL_gridding_slicefirst_double.c').read())
    else:
      print('Using single precission')
      self.prg = Program(ctx, open('./Kernels/OpenCL_gridding_slicefirst_single.c').read())

  def __del__(self):
    if self.radial:
      del self.traj
      del self.dcf
      del self.tmp_fft_array
      del self.cl_kerneltable
      del self.fft2
      del self.fftshift
      del self.deapo_cl
    else:
      del self.tmp_fft_array
      del self.fft2

  def NUFFTH(self,sg,s,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    ### Zero tmp arrays
    self.tmp_fft_array.add_event(self.prg.zero_tmp(queue, (self.tmp_fft_array.size,),None,self.tmp_fft_array.data,wait_for=self.tmp_fft_array.events+wait_for))
    ### Grid k-space
    self.tmp_fft_array.add_event(self.prg.grid_lut(queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,self.tmp_fft_array.data,s.data,self.traj.data, np.int32(self.gridsize), np.int32(sg.shape[2]),
                             self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                             wait_for=wait_for+sg.events+ s.events+self.tmp_fft_array.events))
    ### FFT
    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
#    for event in self.tmp_fft_array.events:
#      event.wait()
    for j in range(s.shape[1]):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*s.shape[0]*s.shape[2]:(j+1)*s.shape[0]*s.shape[2],...],result=self.tmp_fft_array[j*s.shape[0]*s.shape[2]:(j+1)*s.shape[0]*s.shape[2],...],forward=False)[0])
#    for event in self.tmp_fft_array.events:
#      event.wait()
    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
    ### Deapodization and Scaling
    return self.prg.deapo_adj(queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,sg.data,self.tmp_fft_array.data,self.deapo_cl, np.int32(self.tmp_fft_array[idx].shape[-1]),self.DTYPE_real(self.fft_scale)
                              ,wait_for=wait_for+sg.events+s.events+self.tmp_fft_array.events)





  def NUFFT(self,s,sg,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    ### Zero tmp arrays
    self.tmp_fft_array.add_event(self.prg.zero_tmp(queue, (self.tmp_fft_array.size,),None,self.tmp_fft_array.data,wait_for=self.tmp_fft_array.events+wait_for))
    ### Deapodization and Scaling
    self.tmp_fft_array.add_event(self.prg.deapo_fwd(queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,self.tmp_fft_array.data,sg.data,self.deapo_cl, np.int32(self.tmp_fft_array.shape[-1]),self.DTYPE_real(1/self.fft_scale),
                                                    wait_for = wait_for+sg.events+self.tmp_fft_array.events))
    ### FFT
    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
#    for event in self.tmp_fft_array.events:
#      event.wait()
    for j in range(s.shape[1]):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*s.shape[0]*s.shape[2]:(j+1)*s.shape[0]*s.shape[2],...],result=self.tmp_fft_array[j*s.shape[0]*s.shape[2]:(j+1)*s.shape[0]*s.shape[2],...],forward=True)[0])
#    for event in self.tmp_fft_array.events:
#      event.wait()
    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
    ### Resample on Spoke

    return  self.prg.invgrid_lut(queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,s.data,self.tmp_fft_array.data,self.traj.data, np.int32(self.gridsize),
                                np.int32(s.shape[2]),self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                             wait_for = s.events+wait_for+self.tmp_fft_array.events+sg.events)


  def FFTH(self,sg,s,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]

    self.tmp_fft_array.add_event(self.prg.copy(queue,(s.size,),None,self.tmp_fft_array.data,s.data,self.DTYPE_real(1)))
#    self.tmp_fft_array*=self.mask
    for j in range(int(self.fft_shape[0]/self.par_fft)):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=False)[0])
    ### Scaling
    return  (self.prg.copy(queue,(sg.size,),None,sg.data,self.tmp_fft_array.data,self.DTYPE_real(self.fft_scale)))


  def FFT(self,s,sg,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]

    self.tmp_fft_array.add_event(self.prg.copy(queue,(sg.size,),None,self.tmp_fft_array.data,sg.data,self.DTYPE_real(1/self.fft_scale)))
    for j in range(int(self.fft_shape[0]/self.par_fft)):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=True)[0])
#    self.tmp_fft_array*=self.mask
    return (self.prg.copy(queue,(s.size,),None,s.data,self.tmp_fft_array.data,self.DTYPE_real(1)))


  def FFTH_SMS(self,sg,s,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    s.add_event(self.prg.masking(queue,(s.size,),None,s.data,self.mask.data,wait_for=s.events))
    self.tmp_fft_array.add_event(self.prg.copy(queue,(s.size,),None,self.tmp_fft_array.data,(s).data,self.DTYPE_real(1),wait_for=s.events))


    for j in range(s.shape[0]/self.par_fft):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=False)[0])
    return  (self.prg.copy_SMS_adj(queue,(sg.shape[0]*sg.shape[1],sg.shape[-2],sg.shape[-1]),None,sg.data,self.tmp_fft_array.data,self.shift.data,np.int32(self.packs),np.int32(sg.shape[-3]/self.packs),self.DTYPE_real(self.fft_scale)))


  def FFT_SMS(self,s,sg,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    self.tmp_fft_array.add_event(self.prg.copy_SMS_fwd(queue,(sg.shape[0]*sg.shape[1],sg.shape[-2],sg.shape[-1]),None,self.tmp_fft_array.data,sg.data,self.shift.data,np.int32(self.packs),np.int32(sg.shape[-3]/self.packs),self.DTYPE_real(1/self.fft_scale)))

    for j in range(s.shape[0]/self.par_fft):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=True)[0])

    s.add_event(self.prg.copy(queue,(s.size,),None,s.data,self.tmp_fft_array.data,self.DTYPE_real(1),wait_for=self.tmp_fft_array.events))
    return (self.prg.masking(queue,(s.size,),None,s.data,self.mask.data,wait_for=s.events))