#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:45:18 2018

@author: omaier
"""
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

N = 512
Nproj = 5
NScan = 3
NSlice = 5

ga = 111.25*np.pi/180

plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['r','g','b','y','c'])))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
traj = np.mod(np.arange(0,Nproj*NScan)*ga,np.pi)
traj_x = np.outer(np.sin(traj),(np.arange(-N/2,N/2)/N))
traj_y = np.outer(np.cos(traj),(np.arange(-N/2,N/2)/N))
z_dim = np.arange(-NScan/2,NScan/2)
slice_range = np.arange(-NSlice/2,NSlice)/NSlice/2

for k in range(NSlice):
  for j in range(NScan):
    for i in range(Nproj):
        ax.plot(traj_x[i+Nproj*j],traj_y[i+Nproj*j],zs=z_dim[NScan-j-1]+slice_range[k])

ax.text(traj_x[0,-1],traj_y[0,-1],0.5+0.2,"Scan 1")
ax.text(traj_x[Nproj,-1],traj_y[Nproj,-1],-0.5+0.2,"Scan 2")
ax.text(traj_x[2*Nproj,-1],traj_y[2*Nproj,-1],-1.5+0.2,"Scan 3")
ax.legend(("Proj 1", "Proj 2","Proj 3","Proj 4","Proj 5"),loc=0)
ax.set_xlabel('frequency encoding x')
ax.set_ylabel('frequency encoding y')
#ax.set_zlabel('slice encoding')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.quiver(traj_x[-1,-1]+0.15,traj_y[-1,-1]+0.15,0.5+0.2,0,0,-0.5,color='black')
ax.quiver(traj_x[-1,-1]+0.15,traj_y[-1,-1]+0.15,-0.5+0.2,0,0,-0.5,color='black')
ax.text(traj_x[-1,-1]+0.2,traj_y[-1,-1]+0.2,-0.5+0.4,"z-encoding",zdir='z')
ax.quiver(traj_x[-1,-1]+0.15,traj_y[-1,-1]+0.15,-1.5+0.2,0,0,-0.5,color='black')


plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler('color', ['r','r','r','r','r','g','g','g','g','g','b','b','b','b','b','y','y','y','y','y',
                                            'c','c','c','c','c'])))

def update_lines(num, traj_x, traj_y, z_dim,slice_range):
    if num == 0:
      for j in range(0,NScan*Nproj*NSlice):
        lines[j].set_data([],[])
        lines[j].set_3d_properties([])
    scan = int(num/(NSlice*Nproj))
    ind = int(num-scan*NSlice*Nproj)
    proj = int(ind/NSlice)
    islice = ind - proj*Nproj

    lines[num].set_data(traj_x[proj+Nproj*scan],traj_y[proj+Nproj*scan])
    lines[num].set_3d_properties(z_dim[NScan-scan-1]+slice_range[NSlice-islice-1])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.text(traj_x[0,-1],traj_y[0,-1],0.5+0.2,"Scan 1")
ax.text(traj_x[Nproj,-1],traj_y[Nproj,-1],-0.5+0.2,"Scan 2")
ax.text(traj_x[2*Nproj,-1],traj_y[2*Nproj,-1],-1.5+0.2,"Scan 3")
ax.legend(("Proj 1", "Proj 2","Proj 3","Proj 4","Proj 5"),loc=0)
ax.set_xlabel('frequency encoding x')
ax.set_ylabel('frequency encoding y')
#ax.set_zlabel('slice encoding')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.quiver(traj_x[-1,-1]+0.15,traj_y[-1,-1]+0.15,0.5+0.2,0,0,-0.5,color='black')
ax.quiver(traj_x[-1,-1]+0.15,traj_y[-1,-1]+0.15,-0.5+0.2,0,0,-0.5,color='black')
ax.text(traj_x[-1,-1]+0.2,traj_y[-1,-1]+0.2,-0.5+0.4,"z-encoding",zdir='z')
ax.quiver(traj_x[-1,-1]+0.15,traj_y[-1,-1]+0.15,-1.5+0.2,0,0,-0.5,color='black')

lines = [ax.plot([], [], [])[0] for x in range(NScan*Nproj*NSlice)]

# Setting the axes properties
ax.set_xlim3d([-0.5, 0.5])
ax.set_xlabel('X')

ax.set_ylim3d([-0.5, 0.5])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.5, 0.5])
ax.set_zlabel('Z')

ax.set_title('3D Encoding')
# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, NScan*NSlice*Nproj, fargs=(traj_x, traj_y, z_dim,slice_range),
                                   interval=100, blit=False)
line_ani.save("sampling.gif",writer="imagemagick",fps=5)