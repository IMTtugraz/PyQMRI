
# cython: infer_types=True
# cython: profile=False

from __future__ import division
import numpy as np
import time
import gridroutines as NUFFT
import matplotlib.pyplot as plt


plt.ion()
DTYPE = np.complex64
DTYPE_real = np.float32


import pyopencl as cl
import pyopencl.array as clarray
import sys



import matplotlib.pyplot as plt
import multislice_viewer as msv
import matplotlib.gridspec as gridspec
################################################################################
############ Class for handling/building OpenCL Kernels ########################
################################################################################
class Program(object):
    def __init__(self, ctx, code):
        self._cl_prg = cl.Program(ctx, code)
        self._cl_prg.build("-cl-mad-enable -cl-fast-relaxed-math")
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
                self.__dict__[kernel.function_name] = kernel

################################################################################
############ Main Class for the TV Reconstruction ##############################
################################################################################
class Model_Reco:

  def __init__(self,par,ctx,queue,traj=None,dcf=None,trafo=1,ksp_encoding='2D'):
    self.par = par
    self.C = par["C"]
    self.traj = traj
    self.NSlice = par["NSlice"]
    self.NScan = par["NScan"]
    self.dimX = par["dimX"]
    self.dimY = par["dimY"]
    self.NC = par["NC"]
    self.fval_min = 0
    self.fval = 0
    self.ctx = ctx[0]
    self.queue = queue[0]
    self.coil_buf = cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.C.data)
    self.N = par["N"]
    self.Nproj = par["Nproj"]
    self.figure = []
    self.figure_conv = []
    if trafo:
      self.NUFFT = NUFFT.gridding(self.ctx,queue,4,2,par["N"],par["NScan"],(par["NScan"]*par["NC"]*par["NSlice"],par["N"],par["N"]),(1,2),traj.astype(DTYPE),np.require(np.abs(dcf),DTYPE_real,requirements='C'),par["N"],10000,DTYPE,DTYPE_real,radial=trafo)
    else:
      self.NUFFT = NUFFT.gridding(self.ctx,queue,4,2,par["N"],par["NScan"],(par["NScan"]*par["NC"]*par["NSlice"],par["N"],par["N"]),(1,2),traj,dcf,par["N"],1000,DTYPE,DTYPE_real,radial=trafo,mask=par['mask'])

################################################################################
############ OpenCL Kernel definition inline. Could be moved into ##############
############ an external file###################################################
################################################################################
    self.prg = Program(self.ctx, r"""

__kernel void update_r(__global float2 *r, __global float2 *r_, __global float2 *A, __global float2 *A_, __global float2 *res,
                          const float sigma, const float theta, const float lambdainv) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  r[i] = (r_[i]+sigma*((1+theta)*A[i]-theta*A_[i] - res[i]))*lambdainv;
}


  __kernel void update_z1_tv(__global float8 *z_new, __global float8 *z, __global float8 *gx,__global float8 *gx_,
                          const float sigma, const float theta, const float alphainv) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

  float fac = 0.0f;

  z_new[i] = z[i] + sigma*((1+theta)*gx[i]-theta*gx_[i]);
  // reproject
  fac = hypot(fac,hypot(hypot(z_new[i].s0,z_new[i].s1), hypot(hypot(z_new[i].s2,z_new[i].s3),hypot(z_new[i].s4,z_new[i].s5))));
  fac *= alphainv;
  if (fac > 1.0f) z_new[i] /=fac;

}
__kernel void update_primal(__global float2 *u_new, __global float2 *u, __global float2 *Kyk, const float tau) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;
  float norm = 0;


  u_new[i] = u[i]-tau*Kyk[i];

}

__kernel void gradient(__global float8 *grad, __global float2 *u) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;
   // gradient
   grad[i] = (float8)(-u[i],-u[i],-u[i],0.0f,0.0f);
   if (x < Nx-1)
   { grad[i].s01 += u[i+1].s01;}
   else
   { grad[i].s01 = 0.0f;}

   if (y < Ny-1)
   { grad[i].s23 += u[i+Nx].s01;}
   else
   { grad[i].s23 = 0.0f;}
   if (k < NSl-1)
   { grad[i].s45 += u[i+Nx*Ny].s01;}
   else
   { grad[i].s45 = 0.0f;}
}

__kernel void divergence(__global float2 *div, __global float8 *p) {
  size_t Nx = get_global_size(2), Ny = get_global_size(1);
  size_t NSl = get_global_size(0);
  size_t x = get_global_id(2), y = get_global_id(1);
  size_t k = get_global_id(0);
  size_t i = k*Nx*Ny+Nx*y + x;

   // divergence
   float8 val = p[i];
   if (x == Nx-1)
   {
       //real
       val.s0 = 0.0f;
       //imag
       val.s1 = 0.0f;
   }
   if (x > 0)
   {
       //real
       val.s0 -= p[i-1].s0;
       //imag
       val.s1 -= p[i-1].s1;
   }
   if (y == Ny-1)
   {
       //real
       val.s2 = 0.0f;
       //imag
       val.s3 = 0.0f;
   }
   if (y > 0)
   {
       //real
       val.s2 -= p[i-Nx].s2;
       //imag
       val.s3 -= p[i-Nx].s3;
   }
   if (k == NSl-1)
   {
       //real
       val.s4 = 0.0f;
       //imag
       val.s5 = 0.0f;
   }
   if (k > 0)
   {
       //real
       val.s4 -= p[i-Nx*Ny].s4;
       //imag
       val.s5 -= p[i-Nx*Ny].s5;
   }
   div[i] = val.s01+val.s23+val.s45;
}


__kernel void operator_fwd(__global float2 *out, __global float2 *in,
                       __global float2 *coils, const int NCo)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t NSl = get_global_size(0);

  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float2 tmp_in = 0.0f;
  float2 tmp_coil = 0.0f;
  float2 tmp_mul = 0.0f;

  tmp_in = in[k*X*Y+ y*X + x];
  for (int coil=0; coil < NCo; coil++)
  {
    tmp_coil = coils[coil*NSl*X*Y + k*X*Y + y*X + x];
    out[coil*NSl*X*Y+k*X*Y + y*X + x] = (float2)(tmp_in.x*tmp_coil.x-tmp_in.y*tmp_coil.y,
                                              tmp_in.x*tmp_coil.y+tmp_in.y*tmp_coil.x);
  }



}


__kernel void update_Kyk1(__global float2 *out, __global float2 *in,
                       __global float2 *coils, __global float8 *p, const int NCo)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t NSl = get_global_size(0);

  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  size_t i = k*X*Y+X*y + x;

  float2 tmp_in = 0.0f;
  float2 conj_coils = 0.0f;


  float2 sum = (float2) 0.0f;
  for (int coil=0; coil < NCo; coil++)
  {
    conj_coils = (float2) (coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                                  -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);

    tmp_in = in[coil*NSl*X*Y + k*X*Y+ y*X + x];
    sum += (float2)(tmp_in.x*conj_coils.x-tmp_in.y*conj_coils.y,
                                     tmp_in.x*conj_coils.y+tmp_in.y*conj_coils.x);
  }

   // divergence
   float8 val = p[i];
   if (x == X-1)
   {
       //real
       val.s0 = 0.0f;
       //imag
       val.s1 = 0.0f;
   }
   if (x > 0)
   {
       //real
       val.s0 -= p[i-1].s0;
       //imag
       val.s1 -= p[i-1].s1;
   }
   if (y == Y-1)
   {
       //real
       val.s2 = 0.0f;
       //imag
       val.s3 = 0.0f;
   }
   if (y > 0)
   {
       //real
       val.s2 -= p[i-X].s2;
       //imag
       val.s3 -= p[i-X].s3;
   }
   if (k == NSl-1)
   {
       //real
       val.s4 = 0.0f;
       //imag
       val.s5 = 0.0f;
   }
   if (k > 0)
   {
       //real
       val.s4 -= p[i-X*Y].s4;
       //imag
       val.s5 -= p[i-X*Y].s5;
   }
  out[k*X*Y+y*X+x] = sum - (val.s01+val.s23+val.s45);
}

""")
    self.tmp_result = clarray.zeros(self.queue,(self.NScan,self.NC,self.NSlice,self.dimY,self.dimX),DTYPE,"C")
    print("Please Set Parameters, Data and Initial images")


  def operator_forward_full(self, out, x, wait_for=[]):
    self.tmp_result.add_event( self.prg.operator_fwd(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                  self.tmp_result.data, x.data, self.coil_buf, np.int32(self.NC),
                                 wait_for=wait_for+x.events+self.tmp_result.events))
    return  self.NUFFT.fwd_NUFFT(out,self.tmp_result,wait_for=wait_for+self.tmp_result.events)

  def operator_adjoint_full(self, out, x,z, wait_for=[]):
    self.tmp_result.add_event(self.NUFFT.adj_NUFFT(self.tmp_result,x,wait_for=wait_for+x.events))
    return self.prg.update_Kyk1(self.queue, (self.NSlice,self.dimY,self.dimX), None,
                                 out.data, self.tmp_result.data, self.coil_buf, z.data, np.int32(self.NC),
                                 wait_for=self.tmp_result.events+out.events+z.events+wait_for)




  def f_grad(self,grad, u, wait_for=[]):
    return self.prg.gradient(self.queue, u.shape, None, grad.data, u.data,
                wait_for=grad.events + u.events + wait_for)

  def bdiv(self,div, u, wait_for=[]):
    return self.prg.divergence(div.queue, u.shape[:-1], None, div.data, u.data,
                wait_for=div.events + u.events + wait_for)

  def update_primal(self, x_new, x, Kyk, tau, wait_for=[]):
    return self.prg.update_primal(self.queue, x.shape, None, x_new.data, x.data, Kyk.data, np.float32(tau),
                                  wait_for=x_new.events + x.events + Kyk.events+ wait_for
                                  )

  def update_z1_tv(self, z_new, z, gx, gx_, sigma, theta, alpha, wait_for=[]):
    return self.prg.update_z1_tv(self.queue, z.shape[:-1], None, z_new.data, z.data, gx.data, gx_.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/alpha),
                                  wait_for= z_new.events + z.events + gx.events+ gx_.events+ wait_for
                                  )

  def update_r(self, r_new, r, A, A_, res, sigma, theta, lambd, wait_for=[]):
    return self.prg.update_r(self.queue, (self.NScan*self.NC*self.NSlice,self.Nproj,self.N), None, r_new.data, r.data, A.data, A_.data, res.data, np.float32(sigma), np.float32(theta),
                                  np.float32(1/(1+sigma/lambd)),
                                  wait_for= r_new.events + r.events + A.events+ A_.events+ wait_for
                                  )
################################################################################
### Start a 3D TV Reconstruction ###############################################
### input: None, Everything should be set up in ################################
### the initialization of the optimizer ########################################
### output: optimal value of x #################################################
################################################################################
  def execute_3D(self):
################################################################################
### Setup forward and adj NUFFT ################################################
################################################################################
    self.FT = self.NUFFT.fwd_NUFFT
    self.FTH = self.NUFFT.adj_NUFFT
################################################################################
### Allocate Arrays on the Host (CPU) ##########################################
################################################################################
    iters = self.irgn_par["max_iters"]
    self.r = np.zeros_like(self.data,dtype=DTYPE)
    self.z1 = np.zeros(([self.NSlice,self.par["dimX"],self.par["dimY"],4]),dtype=DTYPE)
    self.result = np.zeros((self.par["NSlice"],self.par["dimY"],self.par["dimX"]),dtype=DTYPE)
    self.result = np.copy(self.guess)
    result = np.copy(self.guess)


################################################################################
### Start the TV Reconstricton #################################################
################################################################################
    start = time.time()
    result = self.tv_solve_3D(result,self.data,iters)
    end = time.time()-start

    print("Finished!  Elapsed time: %f seconds" %(end))



################################################################################
### Calculate final value of the primal problem ################################
################################################################################
    b = clarray.zeros(self.queue, self.data.shape,dtype=DTYPE)
    result = clarray.to_device(self.queue,result)
    grad = clarray.to_device(self.queue,np.zeros_like(self.z1))
    grad.add_event(self.f_grad(grad,result,wait_for=grad.events+result.events))
    self.operator_forward_full(b,result)
    result = result.get()
    self.fval= (self.irgn_par["lambd"]/2*np.linalg.norm(self.data - b.get())**2
            +self.irgn_par["gamma"]*np.sum(np.abs(grad.get())))
    self.gn_res = self.fval
    print("-"*80)
    print ("Function value after TV-Reconstruction: %f" %(self.fval/(self.irgn_par["lambd"]*self.NSlice)))

################################################################################
### Set the output/result. could also be returend ##############################
################################################################################
    self.result=result

################################################################################
### Precompute constant terms of the GN linearization step #####################
### input: linearization point x ###############################################
########## numeber of innner iterations iters ##################################
########## Data ################################################################
########## bool to switch between TV (1) and TGV (0) regularization ############
### output: optimal value of x for the inner GN step ###########################
################################################################################
################################################################################
  def tv_solve_3D(self, x,res, iters):


    alpha = self.irgn_par["gamma"]

    e_primal = []
    e_dual = []
    e_gap = []


################################################################################
### Rough guess of Operator Norm ###############################################
################################################################################
    L = np.float32(8)
    tau = np.float32(1/np.sqrt(L))
    tau_new =np.float32(0)
################################################################################
### Send Python NumPy Arrays to the GPU ########################################
### Allocate space for intermediate results ####################################
################################################################################
    x = clarray.to_device(self.queue,x)
    x_new = clarray.zeros_like(x)
    r = clarray.to_device(self.queue,np.zeros_like(self.r))
    r_new  = clarray.zeros_like(r)
    z1 = clarray.to_device(self.queue,np.zeros_like(self.z1))
    z1_new = clarray.zeros_like(z1)
    res = clarray.to_device(self.queue, res.astype(DTYPE))
    Kyk1 = clarray.zeros_like(x)
    Kyk1_new = clarray.zeros_like(x)
    gradx = clarray.zeros_like(z1)
    gradx_xold = clarray.zeros_like(z1)
    Axold = clarray.zeros_like(res)
    Ax = clarray.zeros_like(res)

################################################################################
### Define some algorithm related constants ####################################
################################################################################
    delta = self.irgn_par["lambd"]
    mu = 1/delta
    theta_line = np.float32(1.0)
    beta_line = np.float32(400)
    beta_new = np.float32(0)
    mu_line =np.float32( 0.5)
    delta_line = np.float32(1)
    ynorm = np.float32(0.0)
    lhs = np.float32(0.0)
    primal = np.float32(0.0)
    primal_new = np.float32(0)
    dual = np.float32(0.0)
    gap_min = np.float32(0.0)
    gap = np.float32(0.0)




################################################################################
### Start of the PD Algorithm ##################################################
################################################################################
    Axold.add_event(self.operator_forward_full(Axold,x))
    Kyk1.add_event(self.operator_adjoint_full(Kyk1,r,z1))
    for i in range(iters):

      x_new.add_event(self.update_primal(x_new,x,Kyk1,tau))

      beta_new = beta_line*(1+mu*tau)
      tau_new = tau*np.sqrt(beta_line/beta_new*(1+theta_line))
      beta_line = beta_new

      gradx.add_event(self.f_grad(gradx,x_new,wait_for=gradx.events+x_new.events))
      gradx_xold.add_event(self.f_grad(gradx_xold,x,wait_for=gradx_xold.events+x.events))
      Ax.add_event(self.operator_forward_full(Ax,x_new))


################################################################################
### Line Search for optimal Step Size ##########################################
################################################################################
      while True:
        theta_line = tau_new/tau
        z1_new.add_event(self.update_z1_tv(z1_new,z1,gradx,gradx_xold, beta_line*tau_new, theta_line,alpha))
        r_new.add_event(self.update_r(r_new,r,Ax,Axold,res,beta_line*tau_new,theta_line,self.irgn_par["lambd"]))

        Kyk1_new.add_event(self.operator_adjoint_full(Kyk1_new,r_new,z1_new))

        ynorm = ((clarray.vdot(r_new-r,r_new-r)+clarray.vdot(z1_new-z1,z1_new-z1))**(1/2)).real
        lhs = np.sqrt(beta_line)*tau_new*((clarray.vdot(Kyk1_new-Kyk1,Kyk1_new-Kyk1))**(1/2)).real
        if lhs <= ynorm*delta_line:
            break
        else:
          tau_new = tau_new*mu_line

################################################################################
### Swap updates. No memory allocation needed ##################################
################################################################################
      (Kyk1, Kyk1_new,  Axold, Ax, z1, z1_new, r, r_new) =\
      (Kyk1_new, Kyk1,  Ax, Axold, z1_new, z1, r_new, r)
      tau =  (tau_new)

################################################################################
### Plotting and Calculation of energy values ##################################
################################################################################
      if not np.mod(i,1):
        primal_new= (self.irgn_par["lambd"]/2*clarray.vdot(Axold-res,Axold-res)+alpha*clarray.sum(abs((gradx))) ).real
        dual = (- 1/(2*self.irgn_par["lambd"])*clarray.vdot(r,r) - clarray.vdot(res,r)).real
        gap = np.abs(primal_new - dual)

        e_primal.append(primal_new.get())
        e_dual.append(dual.get())
        e_gap.append(gap.get())

        self.plot_unknowns(x_new.get(),e_primal,e_dual,e_gap,1)

        if i==0:
          gap_min = gap
        if np.abs(primal-primal_new)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"]:
          print("Terminated at iteration %d because the energy decrease in the primal problem was less than %.3e"%(i,abs(primal-primal_new).get()/(self.irgn_par["lambd"]*self.NSlice)))
          self.r = r.get()
          self.z1 = z1.get()
          return x_new.get()
        if (gap > gap_min*self.irgn_par["stag"]) and i>1:
          self.r = r.get()
          self.z1 = z1.get()
          print("Terminated at iteration %d because the method stagnated"%(i))
          return x_new.get()
        if np.abs(gap - gap_min)<(self.irgn_par["lambd"]*self.NSlice)*self.irgn_par["tol"] and i>1:
          self.r = r.get()
          self.z1 = z1.get()
          print("Terminated at iteration %d because the energy decrease in the PD gap was less than %.3e"%(i,abs(gap - gap_min).get()/(self.irgn_par["lambd"]*self.NSlice)))
          return x_new.get()
        primal = primal_new
        gap_min = np.minimum(gap,gap_min)
        sys.stdout.write("Iteration: %d ---- Primal: %f, Dual: %f, Gap: %f \r"%(i,primal.get()/(self.irgn_par["lambd"]*self.NSlice),dual.get()/(self.irgn_par["lambd"]*self.NSlice),gap.get()/(self.irgn_par["lambd"]*self.NSlice)))
        sys.stdout.flush()


      (x, x_new) = (x_new, x)


################################################################################
### Return optimal values ######################################################
################################################################################
    self.r = r.get()
    self.z1 = z1.get()
    return x.get()




################################################################################
### Function taking care of the plotting #######################################
################################################################################
  def plot_unknowns(self,x,primal,dual,gap,scale):
      img = np.abs(x)
      img_min = img.min()
      img_max = img.max()
      [z,y,x] = img.shape
      primal = np.array(primal)
      dual = np.array(dual)
      if not self.figure:
       self.ax = []
       plt.ion()
       self.figure = plt.figure(figsize = (12,6))
       self.figure.subplots_adjust(hspace=0, wspace=0)
       self.gs = gridspec.GridSpec(2,5, width_ratios=[x/(20*z),x/z,1,x/(5*z),x/z],height_ratios=[x/z,1])
       self.figure.tight_layout()
       self.figure.patch.set_facecolor(plt.cm.viridis.colors[0])
       for grid in self.gs:
         self.ax.append(plt.subplot(grid))
         self.ax[-1].axis('off')

       self.img_plot=self.ax[1].imshow((img[int(self.NSlice/2),...]))
       self.img_plot_cor=self.ax[6].imshow((img[:,int(img.shape[1]/2),...]))
       self.img_plot_sag=self.ax[2].imshow(np.flip((img[:,:,int(img.shape[-1]/2)]).T,1))
       self.ax[1].set_title('Magnitude image in a.u.',color='white')
       self.ax[1].set_anchor('SE')
       self.ax[2].set_anchor('SW')
       self.ax[6].set_anchor('NE')
       cax = plt.subplot(self.gs[:,0])
       cbar = self.figure.colorbar(self.img_plot, cax=cax)
       cbar.ax.tick_params(labelsize=12,colors='white')
       cax.yaxis.set_ticks_position('left')
       for spine in cbar.ax.spines:
        cbar.ax.spines[spine].set_color('white')
       for spine in self.ax[4].spines:
         self.ax[4].spines[spine].set_color('white')
       self.ax[4].tick_params(axis='x', colors='white', which='both')
       self.ax[4].tick_params(axis='y', colors='white', which='both')
       self.ax[4].axis('on')
       self.ax[4].patch.set_facecolor('white')
       self.p_en = self.ax[4].plot(np.arange(primal.size)*scale,np.log10(primal),'g',label='Primal Energy')[0]
       self.d_en = self.ax[4].plot(np.arange(primal.size)*scale,np.sign(dual)*np.log10(np.abs(dual)),'r',label='Dual Energy')[0]
       self.ax[4].legend()
       self.ax[4].set_xlabel("Iterations",color='w')
       self.ax[4].set_ylabel("log10 of Energy",color='w')


       plt.draw()
       plt.pause(1e-10)
      else:
       self.img_plot.set_data((img[int(self.NSlice/2),...]))
       self.img_plot_cor.set_data((img[:,int(img.shape[1]/2),...]))
       self.img_plot_sag.set_data(np.flip((img[:,:,int(img.shape[-1]/2)]).T,1))
       self.img_plot.set_clim([img_min,img_max])
       self.img_plot_cor.set_clim([img_min,img_max])
       self.img_plot_sag.set_clim([img_min,img_max])
       self.p_en.set_xdata(np.arange(primal.size)*scale)
       self.p_en.set_ydata(np.log10(primal))
       self.d_en.set_xdata(np.arange(dual.size)*scale)
       self.d_en.set_ydata(np.sign(dual)*np.log10(np.abs(dual)))
       # recompute the ax.dataLim
       self.ax[4].relim()
       # update ax.viewLim using the new dataLim
       self.ax[4].autoscale()
       plt.draw()
       plt.pause(1e-10)

