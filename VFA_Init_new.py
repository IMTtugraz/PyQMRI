import numpy as np

import time
import os
import argparse
import h5py
import sys
from tkinter import filedialog
from tkinter import Tk
import nlinvns_maier as nlinvns



import VFA_model as VFA_model
import goldcomp as goldcomp

import matplotlib.pyplot as plt
import ipyparallel as ipp

from pynfft.nfft import NFFT
import pyopencl as cl

def user_input():
  input_dict = {}
  input_dict["max_iters"] = 300
  input_dict["start_iters"] = 100
  input_dict["max_GN_it"] = 13
  input_dict["lambd"] = 1e2
  input_dict["gamma"] = 1e0
  input_dict["delta"] = 1e-1
  input_dict["display_iterations"] = True
  input_dict["gamma_min"] = 0.18
  input_dict["delta_max"] = 1e2
  input_dict["tol"] = 5e-3
  input_dict["stag"] = 1
  input_dict["delta_inc"] = 2
  input_dict["gamma_dec"] = 0.7
  for name,value in input_dict.items():
    while True:
      print("Editing %s, current value: %f"%(name,value))
      new_val = (input("New value: "))
      if not len(new_val):
         break
      else:
        try:
          new_val = type(value)(new_val)
        except ValueError:
          print("Not an %s" %type(value))
          continue
        else:
          print("%s changed to %f"%(name,new_val))
          input_dict[name] = new_val
          break
  return input_dict

def main(args):
    if args.streamed:
      import Model_Reco_OpenCL_streamed_Kyk2_sep as Model_Reco
    else:
      import Model_Reco_OpenCL as Model_Reco
    DTYPE = np.complex64
    np.seterr(divide='ignore', invalid='ignore')
################################################################################
### Initiate parallel interface ################################################
################################################################################
    c = ipp.Client()
################################################################################
### Select input file ##########################################################
################################################################################
    if args.file == '':
      root = Tk()
      root.withdraw()
      root.update()
      file = filedialog.askopenfilename()
      root.destroy()

    name = file.split('/')[-1]
    file = h5py.File(file)

################################################################################
### Check if file contains all necessary information ###########################
################################################################################
    if args.trafo:
      test_data = ['dcf', 'fa_corr','imag_dat', 'imag_traj', 'real_dat', 'real_traj']
      test_attributes = ['image_dimensions','flip_angle(s)','TR',\
                         'data_normalized_with_dcf']
    else:
      test_data = ['fa_corr','imag_dat', 'real_dat']
      test_attributes = ['image_dimensions','flip_angle(s)','TR']



    for datasets in test_data:
        if not (datasets in list(file.keys())):
            file.close()
            raise NameError("Error: '" + datasets + \
                            "' data was not provided/wrongly named!")
    for attributes in test_attributes:
        if not (attributes in list(file.attrs)):
            file.close()
            raise NameError("Error: '" + attributes + \
                            "' was not provided/wrongly as an attribute!")


################################################################################
### Read Data ##################################################################
################################################################################
    reco_Slices = args.slices
    dimX, dimY, NSlice = (file.attrs['image_dimensions']).astype(int)

    data = file['real_dat'][:,:,int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...].astype(DTYPE)\
       +1j*file['imag_dat'][:,:,int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...].astype(DTYPE)


    if args.trafo:
      traj = file['real_traj'][()].astype(DTYPE) + \
             1j*file['imag_traj'][()].astype(DTYPE)

      dcf = np.array(goldcomp.cmp(traj),dtype=DTYPE)
    else:
      traj=None
      dcf = None

    #Create par struct to store everyting
    class struct:
        pass
    par = struct()

################################################################################
### FA correction ##############################################################
################################################################################

    par.fa_corr = np.flip(file['fa_corr'][()].astype(DTYPE),0)[int(NSlice/2)-int(np.floor((reco_Slices)/2)):int(NSlice/2)+int(np.ceil(reco_Slices/2)),...]
    par.fa_corr[par.fa_corr==0] = 1


    [NScan,NC,reco_Slices,Nproj, N] = data.shape
################################################################################
### Set sequence related parameters ############################################
################################################################################

    par.fa = file.attrs['flip_angle(s)']*np.pi/180
    par.TR          = file.attrs['TR']
    par.NC          = NC
    par.dimY        = dimY
    par.dimX        = dimX
    par.NSlice      = reco_Slices
    par.NScan       = NScan
    par.N = N
    par.Nproj = Nproj

    #### TEST
    par.unknowns_TGV = 2
    par.unknowns_H1 = 0
    par.unknowns = 2



################################################################################
### Coil Sensitivity Estimation ################################################
################################################################################
#% Estimates sensitivities and complex image.
#%(see Martin Uecker: Image reconstruction by regularized nonlinear
#%inversion joint estimation of coil sensitivities and image content)
    class B(Exception):
      pass
    try:
      if not file['Coils'][()].shape[1] >= reco_Slices:
        nlinvNewtonSteps = 6
        nlinvRealConstr  = False

        if args.trafo:
          traj_coil = np.reshape(traj,(NScan*Nproj,N))
          coil_plan = NFFT((dimY,dimX),NScan*Nproj*N)
          coil_plan.x = np.transpose(np.array([np.imag(traj_coil.flatten()),\
                                               np.real(traj_coil.flatten())]))
          coil_plan.precompute()

        par.C = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
        par.phase_map = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)
        result = []
        for i in range(0,(reco_Slices)):
          sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                         %(i))
          sys.stdout.flush()

          ##### RADIAL PART
          if args.trafo:
            combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
            combinedData = np.reshape(combinedData,(NC,NScan*Nproj,N))
            coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
            for j in range(NC):
                coil_plan.f = combinedData[j,:,:]*np.repeat(np.sqrt(dcf),NScan,axis=0)
                coilData[j,:,:] = coil_plan.adjoint()

            combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)
          else:
            combinedData = np.mean(data[:,:,i,...],0)

          dview = c[int(np.floor(i*len(c)/NSlice))]
          result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                          nlinvNewtonSteps, True, nlinvRealConstr))

        for i in range(reco_Slices):
          par.C[:,i,:,:] = result[i].get()[2:,-1,:,:]
          sys.stdout.write("slice %i done \r" \
                         %(i))
          sys.stdout.flush()
          if not nlinvRealConstr:
            par.phase_map[i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
            par.C[:,i,:,:] = par.C[:,i,:,:]* np.exp(1j *\
                 np.angle( result[i].get()[1,-1,:,:]))

            # standardize coil sensitivity profiles
        sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
        if NC == 1:
          par.C = sumSqrC
        else:
          par.C = par.C / np.tile(sumSqrC, (NC,1,1,1))
        del file['Coils']
        file.create_dataset("Coils",par.C.shape,dtype=par.C.dtype,data=par.C)
        file.flush()
      else:
        print("Using precomputed coil sensitivities")
        slices_coils = file['Coils'][()].shape[1]
        par.C = file['Coils'][:,int(slices_coils/2)-int(np.floor((reco_Slices)/2)):int(slices_coils/2)+int(np.ceil(reco_Slices/2)),...]
    except:
      nlinvNewtonSteps = 6
      nlinvRealConstr  = False

      if args.trafo:
        traj_coil = np.reshape(traj,(NScan*Nproj,N))
        coil_plan = NFFT((dimY,dimX),NScan*Nproj*N)
        coil_plan.x = np.transpose(np.array([np.imag(traj_coil.flatten()),\
                                             np.real(traj_coil.flatten())]))
        coil_plan.precompute()

      par.C = np.zeros((NC,reco_Slices,dimY,dimX), dtype=DTYPE)
      par.phase_map = np.zeros((reco_Slices,dimY,dimX), dtype=DTYPE)
      result = []
      for i in range(0,(reco_Slices)):
        sys.stdout.write("Computing coil sensitivity map of slice %i \r" \
                       %(i))
        sys.stdout.flush()

        ##### RADIAL PART
        if args.trafo:
          combinedData = np.transpose(data[:,:,i,:,:],(1,0,2,3))
          combinedData = np.reshape(combinedData,(NC,NScan*Nproj,N))
          coilData = np.zeros((NC,dimY,dimX),dtype=DTYPE)
          for j in range(NC):
              coil_plan.f = combinedData[j,:,:]*np.repeat(np.sqrt(dcf),NScan,axis=0)
              coilData[j,:,:] = coil_plan.adjoint()

          combinedData = np.fft.fft2(coilData,norm=None)/np.sqrt(dimX*dimY)
        else:
          combinedData = np.mean(data[:,:,i,...],0)

        dview = c[int(np.floor(i*len(c)/NSlice))]
        result.append(dview.apply_async(nlinvns.nlinvns, combinedData,
                                        nlinvNewtonSteps, True, nlinvRealConstr))

      for i in range(reco_Slices):
        par.C[:,i,:,:] = result[i].get()[2:,-1,:,:]
        sys.stdout.write("slice %i done \r" \
                       %(i))
        sys.stdout.flush()
        if not nlinvRealConstr:
          par.phase_map[i,:,:] = np.exp(1j * np.angle( result[i].get()[0,-1,:,:]))
          par.C[:,i,:,:] = par.C[:,i,:,:]* np.exp(1j *\
               np.angle( result[i].get()[1,-1,:,:]))

          # standardize coil sensitivity profiles
      sumSqrC = np.sqrt(np.sum((par.C * np.conj(par.C)),0)) #4, 9, 128, 128
      if NC == 1:
        par.C = sumSqrC
      else:
        par.C = par.C / np.tile(sumSqrC, (NC,1,1,1))
      file.create_dataset("Coils",par.C.shape,dtype=par.C.dtype,data=par.C)
      file.flush()
################################################################################
### Standardize data norm ######################################################
################################################################################
    [NScan,NC,NSlice,Nproj, N] = data.shape
    if args.trafo:
      if file.attrs['data_normalized_with_dcf']:
          pass
      else:
          data = data*np.sqrt(dcf)
#### Close File after everything was read
    file.close()
    dscale = np.sqrt(NSlice)*DTYPE(np.sqrt(2*1e3))/(np.linalg.norm(data.flatten()))
    par.dscale = dscale
    data = data* dscale
################################################################################
### generate nFFT  #############################################################
################################################################################

    if args.trafo:
      def nfft(NScan,NC,dimX,dimY,N,Nproj,traj):
        plan = []
        traj_x = np.imag(traj)
        traj_y = np.real(traj)
        for i in range(NScan):
            plan.append([])
            points = np.transpose(np.array([traj_x[i,:,:].flatten(),\
                                            traj_y[i,:,:].flatten()]))
            for j in range(NC):
                plan[i].append(NFFT([dimX,dimY],N*Nproj))
                plan[i][j].x = points
                plan[i][j].precompute()

        return plan

      def nFT(x,plan,dcf,NScan,NC,NSlice,Nproj,N,dimX):
        siz = np.shape(x)
        result = np.zeros((NScan,NC,NSlice,Nproj*N),dtype=DTYPE)
        for i in range(siz[0]):
          for j in range(siz[1]):
            for k in range(siz[2]):
              plan[i][j].f_hat = x[i,j,k,:,:]/dimX
              result[i,j,k,:] = plan[i][j].trafo()*np.sqrt(dcf).flatten()

        return result


      def nFTH(x,plan,dcf,NScan,NC,NSlice,dimY,dimX):
        siz = np.shape(x)
        result = np.zeros((NScan,NC,NSlice,dimY,dimX),dtype=DTYPE)
        for i in range(siz[0]):
          for j in range(siz[1]):
            for k in range(siz[2]):
              plan[i][j].f = x[i,j,k,:,:]*np.sqrt(dcf)
              result[i,j,k,:,:] = plan[i][j].adjoint()

        return result/dimX
      plan = nfft(NScan,NC,dimX,dimY,N,Nproj,traj)

      images= (np.sum(nFTH(data,plan,dcf,NScan,NC,\
                       NSlice,dimY,dimX)*(np.conj(par.C)),axis = 1))
      par.C = np.require(np.transpose(par.C,(0,1,3,2)),requirements='C')
      par.fa_corr = np.require((np.transpose(par.fa_corr,(0,2,1))),requirements='C')
    else:
      images= (np.sum(np.fft.ifft2(data,norm='ortho')*(np.conj(par.C)),axis = 1))




################################################################################
### Create OpenCL Context and Queues ###########################################
################################################################################
    platforms = cl.get_platforms()

    ctx = []
    queue = []
    num_dev = 1#len(platforms[0].get_devices())
    for device in range(num_dev):
      tmp = cl.Context(
              dev_type=cl.device_type.GPU,
              properties=[(cl.context_properties.PLATFORM, platforms[1])])
      ctx.append(tmp)
      queue.append(cl.CommandQueue(tmp,platforms[1].get_devices()[0],properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE | cl.command_queue_properties.PROFILING_ENABLE))
      queue.append(cl.CommandQueue(tmp, platforms[1].get_devices()[0],properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE | cl.command_queue_properties.PROFILING_ENABLE))
      queue.append(cl.CommandQueue(tmp, platforms[1].get_devices()[0],properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE | cl.command_queue_properties.PROFILING_ENABLE))
    if args.trafo:
      opt = Model_Reco.Model_Reco(par,ctx,queue,traj,np.sqrt(dcf),args.trafo)
      opt.data =  data
    else:
      opt = Model_Reco.Model_Reco(par,ctx,queue,None,None,args.trafo)
      opt.data =  data

    if args.type=='3D':
################################################################################
### IRGN - TGV Reco ############################################################
################################################################################
      if "TGV" in args.reg or args.reg=='all':
        result_tgv = []
    ################################################################################
    ### Init forward model and initial guess #######################################
    ################################################################################
        model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,NSlice,Nproj)
        opt.model = model
        opt.dz = 1

    ################################################################################
    ##IRGN Params
        irgn_par = {}
        if args.def_par:
          irgn_par = user_input()
        else:
          irgn_par["max_iters"] = 300
          irgn_par["start_iters"] = 100
          irgn_par["max_GN_it"] = 13
          irgn_par["lambd"] = 1e2
          irgn_par["gamma"] = 1e0
          irgn_par["delta"] = 1e-1
          irgn_par["display_iterations"] = True
          irgn_par["gamma_min"] = 0.24
          irgn_par["delta_max"] = 1e1
          irgn_par["tol"] = 5e-3
          irgn_par["stag"] = 1
          irgn_par["delta_inc"] = 2
          irgn_par["gamma_dec"] = 0.7
        opt.irgn_par = irgn_par

        opt.execute_3D()
        opt.result[:,1,...] = -par.TR/np.log(opt.result[:,1,...])
        result_tgv.append(opt.result)
        plt.close('all')


        res_tgv = opt.gn_res
        res_tgv = np.array(res_tgv)/(irgn_par["lambd"]*NSlice)
  ################################################################################
  #### IRGN - TV referenz ########################################################
  ################################################################################
      if "TV" in args.reg or args.reg=='all':
        result_tv = []
    ################################################################################
    ### Init forward model and initial guess #######################################
    #############################################################re###################
        model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,NSlice,Nproj)
        opt.model = model
        opt.dz = 1
    ################################################################################
    ##IRGN Params
        irgn_par = {}
        if args.def_par:
          irgn_par = user_input()
        else:
          irgn_par["max_iters"] = 300
          irgn_par["start_iters"] = 100
          irgn_par["max_GN_it"] = 13
          irgn_par["lambd"] = 1e2
          irgn_par["gamma"] = 1e0
          irgn_par["delta"] = 1e-1
          irgn_par["display_iterations"] = True
          irgn_par["gamma_min"] = 0.23
          irgn_par["delta_max"] = 1e2
          irgn_par["tol"] = 5e-3
          irgn_par["stag"] = 1
          irgn_par["delta_inc"] = 2
          irgn_par["gamma_dec"] = 0.7
        opt.irgn_par = irgn_par

        opt.execute_3D(1)
        opt.result[:,1,...] = -par.TR/np.log(opt.result[:,1,...])
        result_tv.append(opt.result)
        plt.close('all')

        res_tv = opt.gn_res
        res_tv = np.array(res_tv)/(irgn_par["lambd"]*NSlice)

      del opt
    else:
################################################################################
### IRGN - TGV Reco ############################################################
################################################################################
      if "TGV" in args.reg or args.reg=='all':
        result_tgv = []
    ################################################################################
    ### Init forward model and initial guess #######################################
    ################################################################################
        model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,NSlice,Nproj)
        opt.model = model
        opt.dz = 1

    ################################################################################
    ##IRGN Params
        irgn_par = {}
        if args.def_par:
          irgn_par = user_input()
        else:
          irgn_par["max_iters"] = 300
          irgn_par["start_iters"] = 100
          irgn_par["max_GN_it"] = 13
          irgn_par["lambd"] = 1e2
          irgn_par["gamma"] = 1e0
          irgn_par["delta"] = 1e-1
          irgn_par["display_iterations"] = True
          irgn_par["gamma_min"] = 0.18
          irgn_par["delta_max"] = 1e2
          irgn_par["tol"] = 5e-3
          irgn_par["stag"] = 1
          irgn_par["delta_inc"] = 2
          irgn_par["gamma_dec"] = 0.7
        opt.irgn_par = irgn_par

        opt.execute_2D()
        opt.result[:,1,...] = -par.TR/np.log(opt.result[:,1,...])
        result_tgv.append(opt.result)
        plt.close('all')


        res_tgv = opt.gn_res
        res_tgv = np.array(res_tgv)/(irgn_par["lambd"]*NSlice)
  ################################################################################
  #### IRGN - TV referenz ########################################################
  ################################################################################
      if "TV" in args.reg or args.reg=='all':
        result_tv = []
    ################################################################################
    ### Init forward model and initial guess #######################################
    #############################################################re###################
        model = VFA_model.VFA_Model(par.fa,par.fa_corr,par.TR,images,NSlice,Nproj)
        opt.model = model
    ################################################################################
    ##IRGN Params
        irgn_par = {}
        if args.def_par:
          irgn_par = user_input()
        else:
          irgn_par["max_iters"] = 300
          irgn_par["start_iters"] = 100
          irgn_par["max_GN_it"] = 13
          irgn_par["lambd"] = 1e2
          irgn_par["gamma"] = 1e0
          irgn_par["delta"] = 1e-1
          irgn_par["display_iterations"] = True
          irgn_par["gamma_min"] = 0.23
          irgn_par["delta_max"] = 1e2
          irgn_par["tol"] = 5e-3
          irgn_par["stag"] = 1
          irgn_par["delta_inc"] = 2
          irgn_par["gamma_dec"] = 0.7
        opt.irgn_par = irgn_par

        opt.execute_2D(1)
        opt.result[:,1,...] = -par.TR/np.log(opt.result[:,1,...])
        result_tv.append(opt.result)
        plt.close('all')

        res_tv = opt.gn_res
        res_tv = np.array(res_tv)/(irgn_par["lambd"]*NSlice)

      del opt
###############################################################################
## New .hdf5 save files #######################################################
###############################################################################
    outdir = time.strftime("%Y-%m-%d  %H-%M-%S_MRI_"+args.reg+"_"+args.type+"_"+name[:-3])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    os.makedirs("output/"+ outdir)

    os.chdir("output/"+ outdir)
    f = h5py.File("output_"+name,"w")

    for i in range(len(result_tgv)):
      if "TGV" in args.reg or args.reg=='all':
        f.create_dataset("tgv_full_result_"+str(i),result_tgv[i].shape,\
                                     dtype=DTYPE,data=result_tgv[i])
        f.attrs['res_tgv'] = res_tgv
      if "TV" in args.reg or args.reg=='all':
        f.create_dataset("tv_full_result_"+str(i),result_tv[i].shape,\
                                         dtype=DTYPE,data=result_tv[i])
        f.attrs['res_tv'] = res_tv
      f.attrs['data_norm'] = dscale
      f.flush()
    f.close()

    os.chdir('..')
    os.chdir('..')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1 quantification from VFA data. By default runs 3D regularization for TGV and TV.')
    parser.add_argument('--recon_type', default='3D', dest='type',help='Choose reconstruction type (default: 3D)')
    parser.add_argument('--reg_type', default='TGV', dest='reg',help="Choose regularization type (default: TGV)\
                                                                     options are: TGV, TV, all")
    parser.add_argument('--slices',default=4, dest='slices', type=int, help='Number of reconstructed slices (default=40). Symmetrical around the center slice.')
    parser.add_argument('--def_par', default=0, dest='def_par', type=int, help='Run the script with default (0) or specify your own (1) regularization parameters. ')
    parser.add_argument('--trafo', default=1, dest='trafo',  help='Choos between radial (1, default) and Cartesian (0) sampling. ')
    parser.add_argument('--streamed', default=1, dest='streamed',  help='Enable streaming of large data arrays (>10 slices).')
    parser.add_argument('--data',default='',dest='file',help='Full path to input data. If not provided, a file dialog will open.')
    args = parser.parse_args()

    main(args)