import numpy as np
import phantom as pt
from math import sin,  cos,  pi



def generate_parameter_maps(par, N, M):
    sl_phantom = pt.modified_shepp_logan([N, M, 100])
    slice = np.rot90(sl_phantom[:, :, 50])
    #generate T1 map
    par.T1 = np.copy(slice)
    par.T1[slice>=1] = 800
    tmp1 = 0.18<slice
    tmp2 = 0.22>slice
    par.T1[np.where(tmp1*tmp2)] = 800
    tmp1 = 0.28<slice
    tmp2 = 0.32>slice
    par.T1[np.where(tmp1*tmp2)] = 1200
    tmp1 = slice != 0
    tmp2 = slice < 0.1
    par.T1[np.where(tmp1*tmp2)] =3000
    #generate T2 map
    par.T2 = np.copy(slice)
    par.T2[slice>=1] = 200
    tmp1 = 0.18<slice
    tmp2 = 0.22>slice
    par.T2[np.where(tmp1*tmp2)] = 200
    tmp1 = 0.28<slice
    tmp2 = 0.32>slice
    par.T2[np.where(tmp1*tmp2)] = 600
    tmp1 = slice != 0
    tmp2 = slice < 0.1
    par.T2[np.where(tmp1*tmp2)] =2000
    #generate M0 map
    par.M0 = np.copy(slice)
    par.M0[slice>=1] = 2
    tmp1 = 0.18<slice
    tmp2 = 0.22>slice
    par.M0[np.where(tmp1*tmp2)] = 2
    tmp1 = 0.28<slice
    tmp2 = 0.32>slice
    par.M0[np.where(tmp1*tmp2)] = 1.5
    tmp1 = slice != 0
    tmp2 = slice < 0.1
    par.M0[np.where(tmp1*tmp2)] =4

class Parameter():
    pass



def FLASH( M0f,  T1f,  alphaf,  TRf):
    if (np.size(M0f) != np.size(T1f)):
        print('Error M0 and T1 dont match in size!')
    (N, M) = np.shape(T1f)
    result = np.zeros((N, M))
    np.divide(M0f*sin(alphaf)*(1-np.exp(-np.divide(TRf, T1f))), (1-cos(alphaf)*np.exp(-np.divide(TRf, T1f))), result)
    return np.resize(result, (N, M, 1))
    
 
 
def bSSFP( M0f,  T1f, T2f,  alphaf,  TRf):
    if (np.size(M0f) != np.size(T1f)):
        print('Error M0 and T1 dont match in size!')
    (N, M) = np.shape(T1f)
    result = np.zeros((N, M))
    np.divide(M0f*sin(alphaf)*(1-np.exp(-np.divide(TRf, T1f))),
                    (1-cos(alphaf)*(np.exp(-np.divide(TRf, T1f))-np.exp(-np.divide(TRf, T2f)))-
                    np.multiply(np.exp(-np.divide(TRf, T1f)), np.exp(-np.divide(TRf, T2f)))), result)
    return np.resize(result, (N, M, 1))   

#Taken from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/simulation.py
def generate_birdcage_sensitivities(matrix_size = 256, number_of_coils = 8, relative_radius = 1.5, normalize=True):
    """ Generates birdcage coil sensitivites.
    :param matrix_size: size of imaging matrix in pixels (default ``256``)
    :param number_of_coils: Number of simulated coils (default ``8``)
    :param relative_radius: Relative radius of birdcage (default ``1.5``)
    This function is heavily inspired by the mri_birdcage.m Matlab script in
    Jeff Fessler's IRT package: http://web.eecs.umich.edu/~fessler/code/
    """

    out = np.zeros((matrix_size,matrix_size, number_of_coils),dtype=np.complex64)
    for c in range(0,number_of_coils):
        coilx = relative_radius*np.cos(c*(2*np.pi/number_of_coils))
        coily = relative_radius*np.sin(c*(2*np.pi/number_of_coils))
        coil_phase = -c*(2*np.pi/number_of_coils)

        for y in range(0,matrix_size):
            y_co = float(y-matrix_size/2)/float(matrix_size/2)-coily
            for x in range(0,matrix_size):
                x_co = float(x-matrix_size/2)/float(matrix_size/2)-coilx
                rr = np.sqrt(x_co**2+y_co**2)
                phi = np.arctan2(x_co, -y_co) + coil_phase
                out[y,x,c] =  (1/rr) * np.exp(1j*phi)

    if normalize:
         rss = np.squeeze(np.sqrt(np.sum(abs(out) ** 2, 2)))
         out = out / np.repeat(rss.reshape(matrix_size, matrix_size, 1), number_of_coils, axis=2)

    return out



def generate_phantom_data(N=128, M=128, NC=4, TR = 5,  TR_bSSFP=5,  genBoth = False):
    #N,M image Dimensions, NC number of coils TR and TR_bSSFP
    #Generate Parameter Maps
    params = Parameter()
    params.M0 = []
    params.T1 = []
    params.T2 = []
    generate_parameter_maps(params, N, M)
    M0 = np.float64(params.M0)
    T1 = np.float64(params.T1)
    T2 = np.float64(params.T2)
    #Generate Coil Sensitivities
    coils = generate_birdcage_sensitivities(N, NC)
    #Flip angles for the FLASH sequence, equal to number of Scans
    alpha = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])*pi/180
    #Add Noise to every coil:
    randnoise = 0.00*np.random.randn(N, M, NC)
    coils_noisy = coils+randnoise
    #... and common Noise to the image
    commnoise = 0.00*np.random.randn(N, M, 1)
    #Generate FLASH Data
    data1 = np.zeros((N, M,  NC, np.size(alpha)), dtype =np.complex64)
    for i in range(0, np.size(alpha)):
        data1[:, :, :, i] = (np.multiply(np.repeat(FLASH(M0, T1, alpha[i],  TR)+commnoise, NC, axis=2), (coils_noisy)))
    if (genBoth):
        alpha_bSSFP =  np.array([5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70])*pi/180
        #Add Noise to every coil:
        randnoise = 0.001*np.random.randn(N, M, NC)
        coils_noisy = coils+randnoise
        #... and common Noise to the image
        commnoise = 0.001*np.random.randn(N, M, 1)
        #Generate bSSFP Data
        data2 = np.zeros((N, M,  NC, np.size(alpha_bSSFP)), dtype =np.complex64)
        for i in range(0, np.size(alpha_bSSFP)):
            data2[:, :, :, i] =np.fft.fft2(np.multiply(np.repeat(bSSFP(M0, T1, T2, alpha_bSSFP[i],  TR_bSSFP)+commnoise, NC, axis=2), coils_noisy), axes=(0, 1), norm='ortho')
        return np.concatenate((data1, data2), axis=3)
    else:
        return data1;


    


