import numpy as np
from calckbkernel import calckbkernel
import pyopencl as cl
import pyopencl.array as clarray

import reikna.cluda as cluda
from reikna.fft import FFT, FFTShift
import reikna.transformations as transformations


class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel


class gridding:
  def __init__(self, ctx,queue, kwidth,overgridfactor,G,fft_size,fft_dim,traj,dcf,gridsize,klength=400, DTYPE=np.complex128,DTYPE_real=np.float64):
    (self.kerneltable,self.kerneltable_FT,self.u) = calckbkernel(kwidth,overgridfactor,G,klength)
    self.kernelpoints = self.kerneltable.size
    self.ctx = ctx
    self.queue = queue
    self.cl_kerneltable = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.kerneltable.astype(DTYPE_real).data)
    self.api = cluda.ocl_api()
    self.thr = self.api.Thread(queue)
    self.fft_scale = DTYPE_real(fft_size[-1])
    self.tmp_fft_array = clarray.zeros(self.queue,fft_size,dtype=DTYPE)
    self.tmp_fft_array_r = clarray.zeros(self.queue,fft_size,dtype=DTYPE_real)
    self.tmp_fft_array_i = clarray.zeros(self.queue,fft_size,dtype=DTYPE_real)
    fft = FFT(self.tmp_fft_array, axes=fft_dim)
    fftshift = FFTShift(self.tmp_fft_array,axes=fft_dim)
    self.fftshift = fftshift.compile(self.thr)
    self.fft2 = fft.compile(self.thr)
    self.deapo = 1/self.kerneltable_FT.astype(DTYPE_real)
    self.deapo_cl = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.deapo.data)
    self.kwidth = kwidth/2
    self.dcf = clarray.to_device(self.queue,dcf)
    self.traj = clarray.to_device(self.queue,traj)
    self.gridsize = gridsize
    self.DTYPE = DTYPE
    self.DTYPE_real = DTYPE_real
    if DTYPE==np.complex128:
      print('Using double precission')
      self.prg = Program(self.ctx, r"""

      #pragma OPENCL EXTENSION cl_khr_fp64: enable
      #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
      void AtomicAdd(__global double *val, double delta) {
        union {
          double f;
          ulong  i;
        } old;
        union {
          double f;
          ulong  i;
        } new;
        do {
          old.f = *val;
          new.f = old.f + delta;
        } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
      }

  __kernel void make_complex(__global double2 *out,__global double *re, __global double* im)
  {
    size_t k = get_global_id(0);

    out[k].s0 = re[k];
    out[k].s1 = im[k];

  }
  __kernel void deapo_adj(__global double2 *out, __global double2 *in, __constant double *deapo, const int dim, const double scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;

    out[k*X*Y+y*X+x] = in[k*N*M+n*M+m] * deapo[y] * deapo[x]* scale;

  }
  __kernel void deapo_fwd(__global double2 *out, __global double2 *in, __constant double *deapo, const int dim, const double scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;


    out[k*N*M+n*M+m] = in[k*X*Y+y*X+x] * deapo[y] * deapo[x] * scale;
  }

  __kernel void zero_tmp(__global double *tmp_r, __global double *tmp_i, __global double2 *tmp)
  {
    size_t x = get_global_id(0);

    tmp_r[x] = 0.0f;
    tmp_i[x] = 0.0f;
    tmp[x] = 0.0f;

  }

  __kernel void grid_lut(__global double *sg_r,__global double *sg_i, __global double2 *s, __global double2 *kpos, const int gridsize, const double kwidth, __global double *dcf, __constant double* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;


    double* ptr, pti;
    double2 kdat = s[k+kDim*n+kDim*NDim*scan]*(double2)(dcf[k],dcf[k]);



    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;


    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (double)(gcount1-gridcenter) / (double)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(double)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(double)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;

             AtomicAdd(&(sg_r[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan]),(kern * kdat.s0));
             AtomicAdd(&(sg_i[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan]),(kern * kdat.s1));
  			    }
  			}
  		}
  }


  __kernel void invgrid_lut(__global double2 *s, __global double2 *sg, __global double2 *kpos, const int gridsize, const double kwidth, __global double *dcf, __constant double* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    double kx, ky;
    double fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    double2 tmp_dat = 0.0;


    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (double)(gcount1-gridcenter) / (double)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (double)(gcount2-gridcenter) / (double)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(double)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(double)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;
             tmp_dat += (double2)(kern,kern)*sg[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan];
  			    }
  			}
  		}
    s[k+kDim*n+kDim*NDim*scan]= tmp_dat*(double2)(dcf[k],dcf[k]);
  }
                         """)
    else:
      print('Using single precission')
      self.prg = Program(self.ctx, r"""
      #pragma OPENCL EXTENSION cl_khr_fp64: enable
      #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
 void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

  __kernel void make_complex(__global float2 *out,__global float *re, __global float* im)
  {
    size_t k = get_global_id(0);

    out[k].s0 = re[k];
    out[k].s1 = im[k];

  }
  __kernel void deapo_adj(__global float2 *out, __global float2 *in, __constant float *deapo, const int dim, const float scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;

    out[k*X*Y+y*X+x] = in[k*N*M+n*M+m] * deapo[y]* deapo[x] * scale;

  }
  __kernel void deapo_fwd(__global float2 *out, __global float2 *in, __constant float *deapo, const int dim, const float scale)
  {
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+dim/4;
    size_t M = dim;
    size_t n = y+dim/4;
    size_t N = dim;


    out[k*N*M+n*M+m] = in[k*X*Y+y*X+x] * deapo[y]* deapo[x] * scale;
  }

  __kernel void zero_tmp(__global float *tmp_r, __global float *tmp_i, __global float2 *tmp)
  {
    size_t x = get_global_id(0);

    tmp_r[x] = 0.0f;
    tmp_i[x] = 0.0f;
    tmp[x] = 0.0f;

  }

  __kernel void grid_lut(__global float *sg_r,__global float *sg_i, __global float2 *s, __global float2 *kpos, const int gridsize, const float kwidth, __global float *dcf, __constant float* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;


    float* ptr, pti;
    float2 kdat = s[k+kDim*n+kDim*NDim*scan]*(float2)(dcf[k],dcf[k]);



    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;


    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (float)(gcount1-gridcenter) / (float)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(float)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(float)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;

             AtomicAdd(&(sg_r[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan]),(kern * kdat.s0));
             AtomicAdd(&(sg_i[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan]),(kern * kdat.s1));
  			    }
  			}
  		}
  }


  __kernel void invgrid_lut(__global float2 *s, __global float2 *sg, __global float2 *kpos, const int gridsize, const float kwidth, __global float *dcf, __constant float* kerneltable, const int nkernelpts )
  {
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    float2 tmp_dat = 0.0;


    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    if (ixmin < 0)
      ixmin=0;
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    if (ixmax >= gridsize)
      ixmax=gridsize-1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    if (iymin < 0)
      iymin=0;
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;
    if (iymax >= gridsize)
      iymax=gridsize-1;

  	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
  		dkx = (float)(gcount1-gridcenter) / (float)gridsize  - kx;
  		for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
  			{
          dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;

  			dk = sqrt(dkx*dkx+dky*dky);

  			if (dk < kwidth)
  			    {

  			    fracind = dk/kwidth*(float)(nkernelpts-1);
  			    kernelind = (int)fracind;
  			    fracdk = fracind-(float)kernelind;

  			    kern = kerneltable[(int)kernelind]*(1-fracdk)+
  			    		kerneltable[(int)kernelind+1]*fracdk;
             tmp_dat += (float2)(kern,kern)*sg[gcount1*gridsize+gcount2+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan];
  			    }
  			}
  		}
    s[k+kDim*n+kDim*NDim*scan]= tmp_dat*(float2)(dcf[k],dcf[k]);
  }
                         """)

  def invgrid_lut(self,sg_real,sg_imag,gridsize, kx,ky,s_real,s_imag,nsamples,convwidth,dcf):
    gridcenter = gridsize/2 ## Index of center of grid. */
    kwidth = convwidth/(gridsize)#;	# Width of kernel, in k-space units. */

    for kcount in range(nsamples):
    	# ----- Find limit indices of grid points that current
    	#	 sample may affect (and check they are within grid) ----- */
      ixmin =  int((kx[kcount]-kwidth)*gridsize +gridcenter)
      if (ixmin < 0):
        ixmin=0
      ixmax = int((kx[kcount]+kwidth)*gridsize +gridcenter)+1
      if (ixmax >= gridsize):
        ixmax=gridsize-1
      iymin = int((ky[kcount]-kwidth)*gridsize +gridcenter)
      if (iymin < 0):
        iymin=0
      iymax =  int((ky[kcount]+kwidth)*gridsize +gridcenter)+1
      if (iymax >= gridsize):
        iymax=gridsize-1

      sgrptr = 0
      sgiptr = 0
      for gcount1 in range(ixmin,ixmax):
        dkx = (gcount1-gridcenter) / gridsize  - kx[kcount]
#        col = (gcount1-ixmin)*(iymax-iymin)
        for gcount2 in range(iymin,iymax):#(gcount2 = iymin; gcount2 <= iymax; gcount2++)
          dky = (gcount2-gridcenter) /gridsize - ky[kcount]
          dk = np.sqrt(dkx*dkx+dky*dky)	# k-space separation*/
          if (dk < kwidth):	# sample affects this grid point		*/
    				# Find index in kernel lookup table */
            fracind = dk/kwidth*(self.kernelpoints-1)
            kernelind = int(fracind)
            fracdk = fracind-kernelind;
    				# Linearly interpolate in kernel lut */
            kern = self.kerneltable[int(kernelind)]*(1-fracdk)+ self.kerneltable[int(kernelind+1)]*fracdk;
            sgrptr += kern * sg_real[gcount1,gcount2] * dcf[kcount]
            sgiptr += kern * sg_imag[gcount1,gcount2] * dcf[kcount]

      s_real[kcount] = sgrptr
      s_imag[kcount] = sgiptr
    return s_real+1j*s_imag


  def gridlut(self,kx,ky,s_real,s_imag,nsamples, dcf, sg_real,sg_imag, gridsize, convwidth):
  # Gridding function that uses a lookup table for a circularyl
  #	symmetric convolution kernel, with linear interpolation.
  #	See Notes below */
  #
  ## INPUT/OUTPUT */
  #
  #double *kx;		# Array of kx locations of samples. */
  #double *ky;		# Array of ky locations of samples. */
  #double *s_real;		# Sampled data, real part. */
  #double *s_imag; 	# Sampled data, real part. */
  #int nsamples;		# Number of k-space samples, total. */
  #double *dcf;		# Density compensation factors. */
  #
  #double *sg_real;	# OUTPUT array, real parts of data. */
  #double *sg_imag;	# OUTPUT array, imag parts of data. */
  #int gridsize;		# Number of points in kx and ky in grid. */
  #double convwidth;	# Kernel width, in grid points.	*/
  #double *kerneltable;	# 1D array of convolution kernel values, starting
  #				at 0, and going to convwidth. */
  #int nkernelpts;		# Number of points in kernel lookup-table */
  #
  ##------------------------------------------------------------------
  #	NOTES:
  #
  #	This uses the following formula, which describes the contribution
  #	of each data sample to the value at each grid point:
  #
  #		grid-point value += data value / dcf * kernel(dk)
  #
  #	where:
  #		data value is the complex sampled data value.
  #		dcf is the density compensation factor for the sample point.
  #		kernel is the convolution kernel function.
  #		dk is the k-space separation between sample point and
  #			grid point.
  #
  #	"grid units"  are integers from 0 to gridsize-1, where
  #	the grid represents a k-space of -.5 to .5.
  #
  #	The convolution kernel is passed as a series of values
  #	that correspond to "kernel indices" 0 to nkernelpoints-1.
  #  ------------------------------------------------------------------ */
  #
  #{
  #int kcount;		# Counts through k-space sample locations */
  #int gcount1, gcount2;	# Counters for loops */
  #int col;		# Grid Columns, for faster lookup. */
  #
  #double kwidth;			# Conv kernel width, in k-space units */
  #double dkx,dky,dk;		# Delta in x, y and abs(k) for kernel calc.*/
  #int ixmin,ixmax,iymin,iymax;	# min/max indices that current k may affect*/
  #int kernelind;			# Index of kernel value, for lookup. */
  #double fracdk;			# Fractional part of lookup index. */
  #double fracind;			# Fractional lookup index. */
  #double kern;			# Kernel value, avoid duplicate calculation.*/
  #double *sgrptr;			# Aux. pointer, for loop. */
  #double *sgiptr;			# Aux. pointer, for loop. */
  #int gridsizesq;			# Square of gridsize */
  #int gridcenter;			# Index in output of kx,ky=0 points. */
  #int gptr_cinc;			# Increment for grid pointer. */

    gridcenter = gridsize/2 ## Index of center of grid. */
    kwidth = convwidth/(gridsize)#;	# Width of kernel, in k-space units. */

      # ========= Zero Output Points ========== */

    sg_real[...] = 0
    sg_imag[...] = 0
#    sgrptr = sg_real
#    sgiptr = sg_imag
  #    gridsizesq = gridsize*gridsize
      # ========= Loop Through k-space Samples ========= */
    for kcount in range(nsamples):# (kcount = 0; kcount < nsamples; kcount++)
        	# ----- Find limit indices of grid points that current
        	#	 sample may affect (and check they are within grid) ----- */
          ixmin =  int((kx[kcount]-kwidth)*gridsize +gridcenter)
          if (ixmin < 0):
            ixmin=0
          ixmax = int((kx[kcount]+kwidth)*gridsize +gridcenter)+1
          if (ixmax >= gridsize):
            ixmax=gridsize-1
          iymin = int((ky[kcount]-kwidth)*gridsize +gridcenter)
          if (iymin < 0):
            iymin=0
          iymax =  int((ky[kcount]+kwidth)*gridsize +gridcenter)+1
          if (iymax >= gridsize):
            iymax=gridsize-1
        	  # Increment for grid pointer at end of column to top of next col.*/
          gptr_cinc = gridsize-(iymax-iymin)-1	# 1 b/c at least 1 sgrptr++ */
#          print("ixmin: %i, ixmax: %i, iymin: %i, iymax: %i, gptr_cinc: %i"%(ixmin,ixmax,iymin,iymax,gptr_cinc))
          sgrptr = sg_real[ixmin:ixmax,iymin:iymax].flatten()
          sgiptr = sg_imag[ixmin:ixmax,iymin:iymax].flatten()
          for gcount1 in range(ixmin,ixmax):
            dkx = (gcount1-gridcenter) / gridsize  - kx[kcount]
            col = (gcount1-ixmin)*(iymax-iymin)
            for gcount2 in range(iymin,iymax):#(gcount2 = iymin; gcount2 <= iymax; gcount2++)
              dky = (gcount2-gridcenter) /gridsize - ky[kcount]
              dk = np.sqrt(dkx*dkx+dky*dky)	# k-space separation*/
              if (dk < kwidth):	# sample affects this grid point		*/
        				# Find index in kernel lookup table */
                fracind = dk/kwidth*(self.kernelpoints-1)
                kernelind = int(fracind)
                fracdk = fracind-kernelind;
        				# Linearly interpolate in kernel lut */
                kern = self.kerneltable[int(kernelind)]*(1-fracdk)+ self.kerneltable[int(kernelind+1)]*fracdk;
                sgrptr[col+gcount2-iymin] += kern * s_real[kcount] * dcf[kcount]
                sgiptr[col+gcount2-iymin] += kern * s_imag[kcount] * dcf[kcount]
          sg_real[ixmin:ixmax,iymin:iymax] = np.reshape(sgrptr,(ixmax-ixmin,iymax-iymin))
          sg_imag[ixmin:ixmax,iymin:iymax] = np.reshape(sgiptr,(ixmax-ixmin,iymax-iymin))
    return sg_real+1j*sg_imag

  def adj_NUFFT(self,sg,s,wait_for=None):
    self.prg.zero_tmp(self.queue, (self.tmp_fft_array_r.size,),None,self.tmp_fft_array_r.data,self.tmp_fft_array_i.data,self.tmp_fft_array.data)
    self.prg.grid_lut(self.queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,self.tmp_fft_array_r.data,self.tmp_fft_array_i.data,s.data,self.traj.data, np.int32(self.gridsize),
                             self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                             wait_for = sg.events+s.events)

    self.prg.make_complex(self.queue,(self.tmp_fft_array.size,),None,self.tmp_fft_array.data,self.tmp_fft_array_r.data,self.tmp_fft_array_i.data)
    self.fftshift(self.tmp_fft_array,self.tmp_fft_array)
    self.fft2(self.tmp_fft_array,self.tmp_fft_array,True)
    self.fftshift(self.tmp_fft_array,self.tmp_fft_array)
    self.prg.deapo_adj(self.queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,sg.data,self.tmp_fft_array.data,self.deapo_cl, np.int32(self.tmp_fft_array.shape[-1]),self.DTYPE_real(self.fft_scale))





  def fwd_NUFFT(self,s,sg,wait_for=None):
    self.prg.zero_tmp(self.queue, (self.tmp_fft_array_r.size,),None,self.tmp_fft_array_r.data,self.tmp_fft_array_i.data,self.tmp_fft_array.data)
    self.prg.deapo_fwd(self.queue,(sg.shape[0]*sg.shape[1]*sg.shape[2],sg.shape[3],sg.shape[4]),None,self.tmp_fft_array.data,sg.data,self.deapo_cl, np.int32(self.tmp_fft_array.shape[-1]),self.DTYPE_real(1/self.fft_scale))
    self.fftshift(self.tmp_fft_array,self.tmp_fft_array)
    self.fft2(self.tmp_fft_array,self.tmp_fft_array)
    self.fftshift(self.tmp_fft_array,self.tmp_fft_array)
    return self.prg.invgrid_lut(self.queue,(s.shape[0],s.shape[1]*s.shape[2],s.shape[-2]*self.gridsize),None,s.data,self.tmp_fft_array.data,self.traj.data, np.int32(self.gridsize), self.DTYPE_real(self.kwidth/self.gridsize),self.dcf.data,self.cl_kerneltable,np.int32(self.kernelpoints),
                             wait_for = s.events+sg.events)

