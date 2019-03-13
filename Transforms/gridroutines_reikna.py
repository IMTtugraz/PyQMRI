import numpy as np

import pyopencl as cl
import pyopencl.array as clarray
import reikna.cluda as cluda
from reikna.fft import  FFT#FFTShift
#from gpyfft.fft import FFT

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
    self.fft_shape = (par["NC"]*par["NSlice"],par["N"],par["N"])
    self.traj = par["traj"]
    self.dcf = par["dcf"]
    self.ctx = ctx
    self.queue = queue
    self.api = cluda.ocl_api()
    self.thr = []
    self.thr.append(self.api.Thread(queue[0]))
    if self.radial:
      (self.kerneltable,self.kerneltable_FT,self.u) = calckbkernel(kwidth,overgridfactor,par["N"],klength)
      self.kernelpoints = self.kerneltable.size

      self.fft_scale = DTYPE_real(np.sqrt(np.prod(self.fft_shape[fft_dim[0]:])))
      self.deapo = 1/self.kerneltable_FT.astype(DTYPE_real)
      self.kwidth = kwidth/2
      self.cl_kerneltable = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.kerneltable.astype(DTYPE_real).data)
      self.deapo_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.deapo.data)
      self.dcf = clarray.to_device(self.queue[0],self.dcf)
      self.traj = clarray.to_device(self.queue[0],self.traj)
      self.tmp_fft_array = (clarray.empty(self.queue[0],(self.fft_shape),dtype=DTYPE))
      self.fft2 = []
#      self.fftshift = []

      self.check = np.ones(par["N"],dtype=DTYPE_real)
      self.check[1::2] = -1
      self.check = clarray.to_device(self.queue[0],self.check)
      self.par_fft = int(self.fft_shape[0]*self.fft_shape[1]*self.fft_shape[2])

      fft = FFT(self.tmp_fft_array,axes=fft_dim).compile(self.thr[0])
#      for j in range(len(queue)):
#        fftshift = FFTShift(self.tmp_fft_array,axes=fft_dim)
#        self.fftshift.append(fftshift.compile(self.thr[j]))
#      fft = FFT(ctx,queue[j],self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],out_array=self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],axes=fft_dim)
      self.fft2.append(fft)
      del fft
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
      self.tmp_fft_array = (clarray.empty(self.queue[0],self.fft_shape,dtype=DTYPE))
      self.fft2 = []
      self.par_fft = int(self.fft_shape[0]*self.fft_shape[1]*self.fft_shape[2])
      self.mask = clarray.to_device(self.queue[0],par["mask"])
#      for j in range(len(queue)):
      fft = FFT(self.tmp_fft_array,axes=fft_dim).compile(self.thr[0])
#      fft = FFT(ctx,queue[j],self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],out_array=self.tmp_fft_array[0:int(self.fft_shape[0]/par["NScan"]),...],axes=fft_dim)
      self.fft2.append(fft)
      del fft

    if DTYPE==np.complex128:
      print('Using double precission')
      self.prg = Program(self.ctx,  open('./Kernels/OpenCL_gridding_double.c').read())
    else:
      print('Using single precission')
      self.prg = Program(self.ctx, open('./Kernels/OpenCL_gridding_single_reikna.c').read())
  def __del__(self):
      if self.radial:
        del self.traj
        del self.dcf
        del self.tmp_fft_array
        del self.cl_kerneltable
        del self.fft2
        del self.deapo_cl
        del self.check
        del self.queue
        del self.ctx
      else:
        del self.tmp_fft_array
        del self.fft2
        del self.queue
        del self.ctx

  def NUFFTH(self,sg,s,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]

    ### Zero tmp arrays
    self.tmp_fft_array.add_event(self.prg.zero_tmp(queue, (self.tmp_fft_array.size,),None,self.tmp_fft_array.data,wait_for=s.events+sg.events+self.tmp_fft_array.events+wait_for))
    ### Grid k-space
    self.tmp_fft_array.add_event(self.prg.grid_lut(queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,self.tmp_fft_array.data,s.data,self.traj.data, np.int32(self.gridsize),
                             self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                             wait_for=wait_for+sg.events+ s.events+self.tmp_fft_array.events))
    ### FFT

#    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
    self.tmp_fft_array.add_event(self.prg.fftshift(queue,(self.fft_shape[0],self.fft_shape[1],self.fft_shape[2]),None,self.tmp_fft_array.data,self.check.data))
    for j in range(s.shape[0]):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=False)[0])
    self.tmp_fft_array.add_event(self.prg.fftshift(queue,(self.fft_shape[0],self.fft_shape[1],self.fft_shape[2]),None,self.tmp_fft_array.data,self.check.data))
#    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
    return self.prg.deapo_adj(queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,sg.data,self.tmp_fft_array.data,self.deapo_cl, np.int32(self.tmp_fft_array[idx].shape[-1]),self.DTYPE_real(self.fft_scale)
                                ,wait_for=wait_for+sg.events+s.events+self.tmp_fft_array.events)




  def NUFFT(self,s,sg,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    ### Zero tmp arrays

    self.tmp_fft_array.add_event(self.prg.zero_tmp(queue, (self.tmp_fft_array.size,),None,self.tmp_fft_array.data,wait_for=s.events+sg.events+self.tmp_fft_array.events+wait_for))
    ### Deapodization and Scaling
    self.tmp_fft_array.add_event(self.prg.deapo_fwd(queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,self.tmp_fft_array.data,sg.data,self.deapo_cl, np.int32(self.tmp_fft_array.shape[-1]),self.DTYPE_real(1/self.fft_scale),
                                                    wait_for = wait_for+sg.events+self.tmp_fft_array.events))
    ### FFT
#    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
    self.tmp_fft_array.add_event(self.prg.fftshift(queue,(self.fft_shape[0],self.fft_shape[1],self.fft_shape[2]),None,self.tmp_fft_array.data,self.check.data))
    for j in range(s.shape[0]):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=True)[0])
    self.tmp_fft_array.add_event(self.prg.fftshift(queue,(self.fft_shape[0],self.fft_shape[1],self.fft_shape[2]),None,self.tmp_fft_array.data,self.check.data))
#    self.tmp_fft_array.add_event(self.fftshift[idx](self.tmp_fft_array,self.tmp_fft_array)[0])
    ### Resample on Spoke
    return self.prg.invgrid_lut(queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,s.data,self.tmp_fft_array.data,self.traj.data, np.int32(self.gridsize), self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                               wait_for = s.events+wait_for+self.tmp_fft_array.events)


  def FFTH(self,sg,s,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    s.add_event(self.prg.masking(queue,(s.size,),None,s.data,self.mask.data,wait_for=s.events+self.mask.events))
    for j in range(s.shape[0]):
      self.tmp_fft_array.add_event(self.prg.copy(queue,(self.tmp_fft_array.size,),None,self.tmp_fft_array.data,(s).data,self.DTYPE_real(1),np.int32(j),wait_for=s.events+self.tmp_fft_array.events+s.events))
      for event in self.tmp_fft_array.events:
        event.wait()
      tmp_event = (self.fft2[idx](self.tmp_fft_array,self.tmp_fft_array,inverse=True))
      for event in tmp_event:
        self.tmp_fft_array.add_event(event)
      sg.add_event(self.prg.copy_back(queue,(self.tmp_fft_array.size,),None,sg.data,self.tmp_fft_array.data,self.DTYPE_real(self.fft_scale),
                                 np.int32(j),wait_for=self.tmp_fft_array.events+sg.events))

    return sg.events[-1]


  def FFT(self,s,sg,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
#    print(sg.shape)
#    print(sg.strides)
#    print(s.shape)
#    print(s.strides)
    for j in range(sg.shape[0]):
      self.tmp_fft_array.add_event(self.prg.copy(queue,(self.tmp_fft_array.size,),None,self.tmp_fft_array.data,sg.data,self.DTYPE_real(1/self.fft_scale),np.int32(j),wait_for=sg.events+self.tmp_fft_array.events))
      tmp_event = (self.fft2[idx](self.tmp_fft_array,self.tmp_fft_array,inverse=False))
      for event in tmp_event:
        self.tmp_fft_array.add_event(event)
      s.add_event(self.prg.copy_back(queue,(self.tmp_fft_array.size,),None,s.data,self.tmp_fft_array.data,self.DTYPE_real(1),
                                np.int32(j),wait_for=self.tmp_fft_array.events+s.events))


    return (self.prg.masking(queue,(s.size,),None,s.data,self.mask.data,wait_for=s.events))

  def FFTH_SMS(self,sg,s,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    s.add_event(self.prg.masking(queue,(s.size,),None,s.data,self.mask.data,wait_for=s.events))
    self.tmp_fft_array.add_event(self.prg.copy(queue,(s.size,),None,self.tmp_fft_array.data,(s).data,self.DTYPE_real(1),wait_for=s.events))


    for j in range(s.shape[0]):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=False)[0])
    return  (self.prg.copy_SMS_adj(queue,(sg.shape[0]*sg.shape[1],sg.shape[-2],sg.shape[-1]),None,sg.data,self.tmp_fft_array.data,self.shift.data,np.int32(self.packs),np.int32(sg.shape[-3]/self.packs),self.DTYPE_real(self.fft_scale)))


  def FFT_SMS(self,s,sg,idx=None,wait_for=[]):
    if idx==None:
      idx = 0
      queue=self.queue[0]
    else:
      queue=self.queue[idx]
    self.tmp_fft_array.add_event(self.prg.copy_SMS_fwd(queue,(sg.shape[0]*sg.shape[1],sg.shape[-2],sg.shape[-1]),None,self.tmp_fft_array.data,sg.data,self.shift.data,np.int32(self.packs),np.int32(sg.shape[-3]/self.packs),self.DTYPE_real(1/self.fft_scale)))

    for j in range(s.shape[0]):
      self.tmp_fft_array.add_event(self.fft2[idx].enqueue_arrays(data=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],result=self.tmp_fft_array[j*self.par_fft:(j+1)*self.par_fft,...],forward=True)[0])

    s.add_event(self.prg.copy(queue,(s.size,),None,s.data,self.tmp_fft_array.data,self.DTYPE_real(1),wait_for=self.tmp_fft_array.events))
    return (self.prg.masking(queue,(s.size,),None,s.data,self.mask.data,wait_for=s.events))

