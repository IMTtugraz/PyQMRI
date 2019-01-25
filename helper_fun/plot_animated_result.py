#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:45:18 2018

@author: omaier
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import h5py
from tkinter import filedialog
from tkinter import Tk
import helper_fun.Compute_mask as masking



def run():
  ################################################################################
  ### Select input file ##########################################################
  ################################################################################

  root = Tk()
  root.withdraw()
  root.update()
  file = filedialog.askopenfilename()
  root.destroy()

  file = h5py.File(file)
  #
  #root = Tk()
  #root.withdraw()
  #root.update()
  #file2 = filedialog.askopenfilename()
  #root.destroy()
  #file2 = h5py.File(file2)
  res_name = []
  for name in file.items():
    res_name.append(name[0])


  data = file[res_name[0]][()]
  file.close()
  data = data[-1,:,:]

  #data[:,:5]=0
  ##
  #data2 = file2['tgv_full_result_0'][()]
  #data2 = data2[-1,:,:]

  #data2[:,:5]=0

  Diff = data[1:7]

  DT = np.zeros((data.shape[-3],data.shape[-2],data.shape[-1],3,3),dtype=np.float32)
  DT[...,0,0] = Diff[0,...].real
  DT[...,0,1] = Diff[1,...].real
  DT[...,0,2] = Diff[3,...].real
  DT[...,1,0] = Diff[1,...].real
  DT[...,1,1] = Diff[2,...].real
  DT[...,1,2] = Diff[5,...].real
  DT[...,2,0] = Diff[3,...].real
  DT[...,2,1] = Diff[5,...].real
  DT[...,2,2] = Diff[4,...].real

  DT_eig = np.linalg.eigh(DT)[0]

  FA = np.sqrt(((DT_eig[...,0]-DT_eig[...,1])**2+(DT_eig[...,1]-DT_eig[...,2])**2+(DT_eig[...,0]-DT_eig[...,2])**2)/2*(DT_eig[...,0]**2+DT_eig[...,1]**2+DT_eig[...,2]**2))


  [z,y,x] = data[0].shape
  #M0 = np.abs(data[0])
  #mask = np.ones_like(M0)
  #mask[M0<0.2] = 0
  #mask = masking.compute(M0*mask)

  #M0 = np.abs(data[0])
  #mask[M0<0.5] = 0
  M0 = (np.abs(data[-15:]))#*mask
  M0 = 1/3*(M0[0]+M0[1]+M0[2])
  T1 = (np.abs(FA))#*mask
  #M0 = (np.abs(data[0])-np.abs(data2[0]))/np.abs(data[0])*mask
  #T1 = (np.abs(data[1])-np.abs(data2[1]))/np.abs(data[1])*mask
  #M0 = np.sqrt((np.abs(np.abs(data[0])**2-np.abs(data2[0])**2))/np.abs(data[0])**2*mask)
  #T1 = np.sqrt((np.abs(np.abs(data[1])**2-np.abs(data2[1])**2))/np.abs(data[1])**2*mask)
  M0_min = M0.min()#-1e-2#
  M0_max = M0.max()#*0.51e-2#
  T1_min = T1.min()#10-1e-2#
  T1_max = T1.max()#3000#T1.max()#1301e-2#



  def update_img(num):
    if num >=x:
      num=x-num-1
  #    print(num)
      for i in range(2):
        ax[1+2*i].images[0].set_array(ax[1+2*i].volume[int(np.floor(num/x*z))])
        ax[2+2*i].images[0].set_array(ax[2+2*i].volume[num])
        ax[7+2*i].images[0].set_array(ax[7+2*i].volume[num])
    else:
  #    print(num)
      for i in range(2):
        ax[1+2*i].images[0].set_array(ax[1+2*i].volume[int(num/x*z)])
        ax[2+2*i].images[0].set_array(ax[2+2*i].volume[num])
        ax[7+2*i].images[0].set_array(ax[7+2*i].volume[num])

  # Attaching 3D axis to the figure
  plt.ion()
  ax=[]
  figure = plt.figure(figsize = (12,6))
  figure.subplots_adjust(hspace=0, wspace=0)
  gs = gridspec.GridSpec(2,6, width_ratios=[x/(20*z),x/z,1,x/z,1,x/(20*z)],height_ratios=[x/z,1])
  figure.tight_layout()
  figure.patch.set_facecolor(plt.cm.viridis.colors[0])
  for grid in gs:
    ax.append(plt.subplot(grid))
    ax[-1].axis('off')

  ax[1].volume=M0
  ax[2].volume=np.flip(np.transpose(M0,(2,1,0)),-1)
  ax[7].volume=np.transpose(M0,(1,0,2))
  M0_plot=ax[1].imshow((M0[int(z/2),...]))
  M0_plot_cor=ax[7].imshow((M0[:,int(M0.shape[1]/2),...]))
  M0_plot_sag=ax[2].imshow(np.flip((M0[:,:,int(M0.shape[-1]/2+3)]).T,1))
  ax[1].set_title('Proton Density in a.u.',color='white')
  ax[1].set_anchor('SE')
  ax[2].set_anchor('SW')
  ax[7].set_anchor('NW')
  cax = plt.subplot(gs[:,0])
  cbar = figure.colorbar(M0_plot, cax=cax)
  cbar.ax.tick_params(labelsize=12,colors='white')
  cax.yaxis.set_ticks_position('left')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  M0_plot.set_clim([M0_min,M0_max])
  M0_plot_cor.set_clim([M0_min,M0_max])
  M0_plot_sag.set_clim([M0_min,M0_max])

  ax[3].volume=T1
  ax[4].volume=np.flip(np.transpose(T1,(2,1,0)),-1)
  ax[9].volume=np.transpose(T1,(1,0,2))
  T1_plot=ax[3].imshow((T1[int(z/2),...]))
  T1_plot_cor=ax[9].imshow((T1[:,int(T1.shape[1]/2),...]))
  T1_plot_sag=ax[4].imshow(np.flip((T1[:,:,int(T1.shape[-1]/2+3)]).T,1))
  ax[3].set_title('T1 in  ms',color='white')
  ax[3].set_anchor('SE')
  ax[4].set_anchor('SW')
  ax[9].set_anchor('NW')
  cax = plt.subplot(gs[:,5])
  cbar = figure.colorbar(T1_plot, cax=cax)
  cbar.ax.tick_params(labelsize=12,colors='white')
  for spine in cbar.ax.spines:
    cbar.ax.spines[spine].set_color('white')
  plt.draw()
  plt.pause(1e-10)
  T1_plot.set_clim([T1_min,T1_max])
  T1_plot_sag.set_clim([T1_min,T1_max])
  T1_plot_cor.set_clim([T1_min,T1_max])
  ax = np.array(ax)

  #plt.savefig("full_gpu.svg",facecolor=plt.cm.viridis.colors[0])
  #ax_trans.volume = np.abs(data[-1,1])
  #ax_trans.index = 0
  #ax_trans.imshow(ax_trans.volume[ax_trans.index],animated=True)
  #ax_cor.volume = np.abs(np.transpose(data[-1,1],(1,0,2)))
  #ax_cor.index = 0
  #ax_cor.imshow(ax_cor.volume[ax_cor.index],animated=True)
  #ax_sag.volume = np.abs(np.transpose(data[-1,1],(2,0,1)))
  #ax_sag.index = 0
  #ax_sag.imshow(ax_sag.volume[ax_sag.index],animated=True)




  #ax.set_title('3D Encoding')
  # Creating the Animation object

  line_ani = animation.FuncAnimation(figure, update_img, x, interval=100, blit=False)
  #line_ani.save("3D_reco.gif",writer="imagemagick",dpi=70,fps=20,savefig_kwargs={'facecolor':plt.cm.viridis.colors[0]})
  
if __name__ == '__main__':
  run()