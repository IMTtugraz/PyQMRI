Prerequests on the .h5 file
============================
The toolbox expects a .h5 file with a certain structure. 

* kspace data (assumed to be 5D for VFA) and passed as:

  - real_dat (Scans, Coils, Slices, Projections, Samples)
  - imag_dat (Scans, Coils, Slices, Projections, Samples)
  
  If radial sampling is used the trajectory is expected to be:
  
  - real_traj (Scans, Projections, Samples)
  - imag_traj (Scans, Projections, Samples)

  | Density compensation is performed internally assuming a simple ramp.
  | For Cartesian data Projections and Samples are replaced by ky and kx encodings points and no trajectory is needed.  
  | Data is assumed to be 2D stack-of-stars, i.e. already Fourier transformed along the fully sampled z-direction.

* flip angle correction (optional) can be passed as:

  - fa_corr (Scans, Coils, Slices, dimY, dimX)

* The image dimension for the full dataset is passed as attribute consiting of:

  - image_dimensions = (dimX, dimY, NSlice)

* Parameters specific to the used model (e.g. TR or flip angle) need to be set as attributes e.g.:

  - TR = 5.38
  - flip_angle(s) = (1,3,5,7,9,11,13,15,17,19)

The specific structure is determined according to the Model file.
    
If predetermined coil sensitivity maps are available they can be passed as complex dataset, which can saved bedirectly using Python. Matlab users would need to write/use low level hdf5 functions to save a complex array to .h5 file. Coil sensitivities are assumed to have the same number of slices as the original volume and are intesity normalized. The corresponding .h5 entry is named "Coils". If no "Coils" parameter is found or the number of "Coil" slices is less than the number of reconstructed slices, the coil sensitivities are determined using the NLINV_ algorithm and saved into the file. 

.. _NLINV: https://doi.org/10.1002/mrm.21691
