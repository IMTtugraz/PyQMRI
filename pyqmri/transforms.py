#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for different FFT operators."""
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from gpyfft.fft import FFT
from pkg_resources import resource_filename
from pyqmri._helper_fun._calckbkernel import calckbkernel
from pyqmri._helper_fun import CLProgram as Program


class PyOpenCLnuFFT():
    """Base class for FFT calculation.

    This class serves as the base class for all FFT object used in
    the varous optimization algorithms. It provides a factory method
    to generate a FFT object based on the input.

    Parameters
    ----------
        ctx : PyOpenCL.Context
          The context for the PyOpenCL computations.
        queue : PyOpenCL.Queue
          The computation Queue for the PyOpenCL kernels.
        fft_dim : tuple of int
          The dimensions to take the fft over
        DTYPE : Numpy.dtype
          The comlex precision type. Currently complex64 is used.
        DTYPE_real : Numpy.dtype
          The real precision type. Currently float32 is used.

    Attributes
    ----------
      DTYPE : Numpy.dtype
        The comlex precision type. Currently complex64 is used.
      DTYPE_real : Numpy.dtype
        The real precision type. Currently float32 is used.
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      prg : PyOpenCL.Program
        The PyOpenCL Program Object containing the compiled kernels.
      fft_dim : tuple of int
        The dimensions to take the fft over
    """

    def __init__(self, ctx, queue, fft_dim, DTYPE, DTYPE_real):
        self.DTYPE = DTYPE
        self.DTYPE_real = DTYPE_real
        self.ctx = ctx
        self.queue = queue
        self.prg = None
        self.fft_dim = fft_dim

    @staticmethod
    def create(ctx,
               queue,
               par,
               kwidth=5,
               klength=200,
               DTYPE=np.complex64,
               DTYPE_real=np.float32,
               radial=False,
               SMS=False,
               streamed=False):
        """FFT factory method.

        Based on the inputs this method decides which FFT object should be
        returned.

        Parameters
        ----------
          ctx : PyOpenCL.Context
            The context for the PyOpenCL computations.
          queue : PyOpenCL.Queue
            The computation Queue for the PyOpenCL kernels.
          par : dict
            A python dict containing the necessary information to setup the
            object. Needs to contain the number of slices (NSlice), number of
            scans (NScan), image dimensions (dimX, dimY), number of coils (NC),
            sampling points (N) and read outs (NProj) a PyOpenCL queue (queue)
            and the complex coil sensitivities (C).
          kwidth : int, 5
            The width of the sampling kernel for regridding of non-uniform
            kspace samples.
          klength : int, 200
            The length of the kernel lookup table which samples the contineous
            gridding kernel.
          DTYPE : Numpy.dtype, numpy.complex64
            The comlex precision type. Currently complex64 is used.
          DTYPE_real : Numpy.dtype, numpy.float32
            The real precision type. Currently float32 is used.
          radial : bool, False
            Switch for Cartesian (False) and non-Cartesian (True) FFT.
          SMS : bool, False
            Switch between Simultaneous Multi Slice reconstruction (True) and
            simple slice by slice reconstruction.
          streamed : bool, False
            Switch between normal reconstruction in one big block versus
            streamed reconstruction of smaller blocks.

        Returns
        -------
          PyOpenCLnuFFT object:
            The setup FFT object.

        Raises
        ------
          AssertionError:
            If the Combination of passed flags to choose the
            FFT aren't compatible with each other. E.g.: Radial and SMS True.
        """
        if not streamed:
            if radial is True and SMS is False:
                obj = PyOpenCLRadialNUFFT(
                    ctx,
                    queue,
                    par,
                    kwidth=kwidth,
                    klength=klength,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            elif SMS is True and radial is False:
                obj = PyOpenCLSMSNUFFT(
                    ctx,
                    queue,
                    par,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            elif SMS is False and radial is False:
                obj = PyOpenCLCartNUFFT(
                    ctx,
                    queue,
                    par,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            else:
                raise AssertionError("Combination of Radial "
                                     "and SMS not allowed")
            if DTYPE == np.complex128:
                print('Using double precision')
                file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_gridding_double.c'))
                obj.prg = Program(
                    obj.ctx,
                    file.read())
            else:
                print('Using single precision')
                file = open(
                    resource_filename(
                        'pyqmri', 'kernels/OpenCL_gridding_single.c'))
                obj.prg = Program(
                    obj.ctx,
                    file.read())
        else:
            if radial is True and SMS is False:
                obj = PyOpenCLRadialNUFFTStreamed(
                    ctx,
                    queue,
                    par,
                    kwidth=kwidth,
                    klength=klength,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            elif SMS is True and radial is False:
                obj = PyOpenCLSMSNUFFTStreamed(
                    ctx,
                    queue,
                    par,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            elif SMS is False and radial is False:
                obj = PyOpenCLCartNUFFTStreamed(
                    ctx,
                    queue,
                    par,
                    DTYPE=DTYPE,
                    DTYPE_real=DTYPE_real)
            else:
                raise AssertionError("Combination of Radial "
                                     "and SMS not allowed")
            if DTYPE == np.complex128:
                print('Using double precision')
                file = open(
                    resource_filename(
                        'pyqmri',
                        'kernels/OpenCL_gridding_slicefirst_double.c'))
                obj.prg = Program(
                    obj.ctx,
                    file.read())
            else:
                print('Using single precision')
                file = open(
                    resource_filename(
                        'pyqmri',
                        'kernels/OpenCL_gridding_slicefirst_single.c'))
                obj.prg = Program(
                    obj.ctx,
                    file.read())
        file.close()
        return obj


class PyOpenCLRadialNUFFT(PyOpenCLnuFFT):
    """Non-uniform FFT object.

    This class performs the non-uniform FFT (NUFFT) operation. Linear
    interpolation of a sampled gridding kernel is used to regrid points
    from the non-cartesian grid back on the cartesian grid.

    Parameters
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      kwidth : int
        The width of the sampling kernel for regridding of non-uniform
        kspace samples.
      klength : int
        The length of the kernel lookup table which samples the contineous
        gridding kernel.
      DTYPE : Numpy.dtype
        The comlex precision type. Currently complex64 is used.
      DTYPE_real : Numpy.dtype
        The real precision type. Currently float32 is used.

    Attributes
    ----------
      traj : PyOpenCL.Array
        The comlex sampling trajectory
      dcf : PyOpenCL.Array
        The densitiy compenation function
      ogf (float):
        The overgriddingfactor for non-cartesian k-spaces.
      fft_shape : tuple of ints
        3 dimensional tuple. Dim 0 containts all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale : float32
        The scaling factor to achieve a good adjointness of the forward and
        backward FFT.
      cl_kerneltable (PyOpenCL.Buffer):
        The gridding lookup table as read only Buffer
      cl_deapo (PyOpenCL.Buffer):
        The deapodization lookup table as read only Buffer
      par_fft : int
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft : gpyfft.fft.FFT
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused in each iterations, iterationg over
        all scans to keep the memory footprint low.
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            ctx,
            queue,
            par,
            kwidth=5,
            klength=200,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, par["fft_dim"], DTYPE, DTYPE_real)
        self.ogf = par["N"]/par["dimX"]
        self.fft_shape = (
            par["NScan"] *
            par["NC"] *
            par["NSlice"],
            int(par["dimY"]*self.ogf),
            int(par["dimX"]*self.ogf))
        self.fft_scale = DTYPE_real(
            np.sqrt(np.prod(self.fft_shape[self.fft_dim[0]:])))

        (kerneltable, kerneltable_FT) = calckbkernel(
            kwidth, self.ogf, par["N"], klength)

        deapo = 1 / kerneltable_FT.astype(DTYPE_real)

        self.cl_kerneltable = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=kerneltable.astype(DTYPE_real).data)
        self.cl_deapo = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=deapo.data)
        self.dcf = clarray.to_device(self.queue, par["dcf"])
        self.traj = clarray.to_device(self.queue, par["traj"])
        self._tmp_fft_array = (
            clarray.empty(
                self.queue,
                (self.fft_shape),
                dtype=DTYPE))
        self.par_fft = int(self.fft_shape[0] / par["NScan"])
        self.fft = FFT(ctx, queue, self._tmp_fft_array[
            0:self.par_fft, ...],
                       out_array=self._tmp_fft_array[
                           0:self.par_fft, ...],
                       axes=self.fft_dim)

        self._kernelpoints = kerneltable.size
        self._kwidth = kwidth / 2
        self._check = np.ones(par["N"], dtype=DTYPE_real)
        self._check[1::2] = -1
        self._check = clarray.to_device(self.queue, self._check)
        self._gridsize = par["N"]

    def __del__(self):
        """Explicitly delete OpenCL Objets."""
        del self.traj
        del self.dcf
        del self._tmp_fft_array
        del self.cl_kerneltable
        del self.cl_deapo
        del self._check
        del self.queue
        del self.ctx
        del self.prg
        del self.fft

    def FFTH(self, sg, s, wait_for=None, scan_offset=0):
        """Perform the inverse (adjoint) NUFFT operation.

        Parameters
        ----------
          sg : PyOpenCL.Array
            The complex image data.
          s : PyOpenCL.Array
            The non-uniformly gridded k-space
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.
          scan_offset : int, 0
            Offset compared to the first acquired scan.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        # Zero tmp arrays
        self._tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self._tmp_fft_array.size,
                 ),
                None,
                self._tmp_fft_array.data,
                wait_for=s.events +
                sg.events +
                self._tmp_fft_array.events +
                wait_for))
        # Grid k-space
        self._tmp_fft_array.add_event(
            self.prg.grid_lut(
                self.queue,
                (s.shape[0], s.shape[1] * s.shape[2],
                 s.shape[-2] * self._gridsize),
                None,
                self._tmp_fft_array.data,
                s.data,
                self.traj.data,
                np.int32(self._gridsize),
                self.DTYPE_real(self._kwidth / self._gridsize),
                self.dcf.data,
                self.cl_kerneltable,
                np.int32(self._kernelpoints),
                np.int32(scan_offset),
                wait_for=(wait_for + sg.events +
                          s.events + self._tmp_fft_array.events)))
        # FFT
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))
        for j in range(s.shape[0]):
            self._tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self._tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self._tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=False)[0])
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))
        return self.prg.deapo_adj(
            self.queue,
            (sg.shape[0] * sg.shape[1] *
             sg.shape[2], sg.shape[3], sg.shape[4]),
            None,
            sg.data,
            self._tmp_fft_array.data,
            self.cl_deapo,
            np.int32(self._tmp_fft_array.shape[-1]),
            self.DTYPE_real(self.fft_scale),
            self.DTYPE_real(self.ogf),
            wait_for=(wait_for + sg.events + s.events +
                      self._tmp_fft_array.events))

    def FFT(self, s, sg, wait_for=None, scan_offset=0):
        """Perform the forward NUFFT operation.

        Parameters
        ----------
          s : PyOpenCL.Array
            The non-uniformly gridded k-space.
          sg : PyOpenCL.Array
            The complex image data.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.
          scan_offset : int, 0
            Offset compared to the first acquired scan.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        # Zero tmp arrays
        self._tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self._tmp_fft_array.size,
                 ),
                None,
                self._tmp_fft_array.data,
                wait_for=s.events +
                sg.events +
                self._tmp_fft_array.events +
                wait_for))
        # Deapodization and Scaling
        self._tmp_fft_array.add_event(
            self.prg.deapo_fwd(
                self.queue,
                (sg.shape[0] * sg.shape[1] * sg.shape[2],
                 sg.shape[3], sg.shape[4]),
                None,
                self._tmp_fft_array.data,
                sg.data,
                self.cl_deapo,
                np.int32(self._tmp_fft_array.shape[-1]),
                self.DTYPE_real(1 / self.fft_scale),
                self.DTYPE_real(self.ogf),
                wait_for=wait_for + sg.events + self._tmp_fft_array.events))
        # FFT
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))

        for j in range(s.shape[0]):
            self._tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self._tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    result=self._tmp_fft_array[
                        j * self.par_fft:(j + 1) * self.par_fft, ...],
                    forward=True)[0])
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))
        # Resample on Spoke
        return self.prg.invgrid_lut(
            self.queue,
            (s.shape[0], s.shape[1] * s.shape[2], s.shape[-2] *
             self._gridsize),
            None,
            s.data,
            self._tmp_fft_array.data,
            self.traj.data,
            np.int32(self._gridsize),
            self.DTYPE_real(self._kwidth / self._gridsize),
            self.dcf.data,
            self.cl_kerneltable,
            np.int32(self._kernelpoints),
            np.int32(scan_offset),
            wait_for=s.events + wait_for + self._tmp_fft_array.events)


class PyOpenCLCartNUFFT(PyOpenCLnuFFT):
    """Cartesian FFT object.

    This class performs the FFT operation.

    Parameters
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      DTYPE : Numpy.dtype
        The comlex precision type. Currently complex64 is used.
      DTYPE_real : Numpy.dtype
        The real precision type. Currently float32 is used.

    Attributes
    ----------
      fft_shape : tuple of ints
        3 dimensional tuple. Dim 0 containts all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale : float32
        The scaling factor to achieve a good adjointness of the forward and
        backward FFT.
      par_fft : int
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft : gpyfft.fft.FFT
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused in each iterations, iterationg over
        all scans to keep the memory footprint low.
      mask : PyOpenCL.Array
        The undersampling mask for the Cartesian grid.
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            ctx,
            queue,
            par,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, par["fft_dim"], DTYPE, DTYPE_real)
        self.fft_shape = (
            par["NScan"] *
            par["NC"] *
            par["NSlice"],
            par["dimY"],
            par["dimX"])

        if par["fft_dim"] is not None:
            self.fft_scale = DTYPE_real(
                np.sqrt(np.prod(self.fft_shape[self.fft_dim[0]:])))
            self._tmp_fft_array = (
                clarray.empty(
                    self.queue,
                    self.fft_shape,
                    dtype=DTYPE))
            self.par_fft = int(self.fft_shape[0] / par["NScan"] / par["NC"])
            self.mask = clarray.to_device(self.queue, par["mask"])
            self.fft = FFT(ctx, queue, self._tmp_fft_array[
                0:self.par_fft, ...],
                           out_array=self._tmp_fft_array[
                               0:self.par_fft, ...],
                           axes=self.fft_dim)

    def __del__(self):
        """Explicitly delete OpenCL Objets."""
        if self.fft_dim is not None:
            del self._tmp_fft_array
            del self.fft
            del self.mask
        del self.queue
        del self.ctx
        del self.prg

    def FFTH(self, sg, s, wait_for=None, scan_offset=0):
        """Perform the inverse (adjoint) FFT operation.

        Parameters
        ----------
          sg : PyOpenCL.Array
            The complex image data.
          s : PyOpenCL.Array
            The uniformly gridded k-space
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.
          scan_offset : int, 0
            Offset compared to the first acquired scan.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.maskingcpy(
                    self.queue,
                    (self._tmp_fft_array.shape),
                    None,
                    self._tmp_fft_array.data,
                    s.data,
                    self.mask.data,
                    wait_for=s.events+self._tmp_fft_array.events+wait_for))
            for j in range(np.prod(s.shape[0:2])):
                self._tmp_fft_array.add_event(
                    self.fft.enqueue_arrays(
                        data=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        result=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        forward=False)[0])
            return (
                self.prg.copy(
                    self.queue,
                    (sg.size,
                     ),
                    None,
                    sg.data,
                    self._tmp_fft_array.data,
                    self.DTYPE_real(
                        self.fft_scale)))

        return self.prg.copy(
                    self.queue,
                    (s.size,
                     ),
                    None,
                    sg.data,
                    s.data,
                    self.DTYPE_real(1),
                    wait_for=s.events+sg.events+wait_for)

    def FFT(self, s, sg, wait_for=None, scan_offset=0):
        """Perform the forward FFT operation.

        Parameters
        ----------
          s : PyOpenCL.Array
            The uniformly gridded k-space.
          sg : PyOpenCL.Array
            The complex image data.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.
          scan_offset : int, 0
            Offset compared to the first acquired scan.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.copy(
                    self.queue,
                    (sg.size,
                     ),
                    None,
                    self._tmp_fft_array.data,
                    sg.data,
                    self.DTYPE_real(
                        1 /
                        self.fft_scale),
                    wait_for=s.events+self._tmp_fft_array.events+wait_for))

            for j in range(np.prod(s.shape[0:2])):
                self._tmp_fft_array.add_event(
                    self.fft.enqueue_arrays(
                        data=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        result=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        forward=True)[0])
            return (
                self.prg.maskingcpy(
                    self.queue,
                    (self._tmp_fft_array.shape),
                    None,
                    s.data,
                    self._tmp_fft_array.data,
                    self.mask.data,
                    wait_for=s.events+self._tmp_fft_array.events))

        return self.prg.copy(
                    self.queue,
                    (sg.size,
                     ),
                    None,
                    s.data,
                    sg.data,
                    self.DTYPE_real(1),
                    wait_for=s.events+sg.events+wait_for)


class PyOpenCLSMSNUFFT(PyOpenCLnuFFT):
    """Cartesian FFT-SMS object.

    This class performs the FFT operation assuming a SMS acquisition.

    Parameters
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      DTYPE : Numpy.dtype
        The comlex precision type. Currently complex64 is used.
      DTYPE_real : Numpy.dtype
        The real precision type. Currently float32 is used.

    Attributes
    ----------
      fft_shape : tuple of ints
        3 dimensional tuple. Dim 0 containts all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale : float32
        The scaling factor to achieve a good adjointness of the forward and
        backward FFT.
      par_fft : int
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft : gpyfft.fft.FFT
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused in each iterations, iterationg over
        all scans to keep the memory footprint low.
      mask : PyOpenCL.Array
        The undersampling mask for the Cartesian grid.
      packs : int
        The distance between the slices
      MB : int
        The multiband factor
      shift : PyOpenCL.Array
        The vector pixel shifts used in the fft computation.
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            ctx,
            queue,
            par,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, par["fft_dim"], DTYPE, DTYPE_real)
        self.fft_shape = (
            par["NScan"] *
            par["NC"] *
            par["NSlice"],
            par["dimY"],
            par["dimX"])

        self.packs = int(par["packs"])
        self.MB = int(par["MB"])
        self.shift = clarray.to_device(
            self.queue, par["shift"].astype(DTYPE_real))

        self.par_fft = int(self.fft_shape[0] / par["NScan"])

        self.mask = clarray.to_device(self.queue, par["mask"])
        if self.fft_dim is not None:
            self.fft_scale = DTYPE_real(
                np.sqrt(np.prod(self.fft_shape[self.fft_dim[0]:])))
            self._tmp_fft_array = (
                clarray.empty(
                    self.queue,
                    self.fft_shape,
                    dtype=DTYPE))
            self.fft = FFT(ctx, queue, self._tmp_fft_array[
                0:self.par_fft, ...],
                           out_array=self._tmp_fft_array[
                               0:self.par_fft, ...],
                           axes=self.fft_dim)

    def __del__(self):
        """Explicitly delete OpenCL Objets."""
        if self.fft_dim is not None:
            del self._tmp_fft_array
            del self.fft
        del self.mask
        del self.queue
        del self.ctx
        del self.prg

    def FFTH(self, sg, s, wait_for=None, scan_offset=0):
        """Perform the inverse (adjoint) FFT operation.

        Parameters
        ----------
          sg : PyOpenCL.Array
            The complex image data.
          s : PyOpenCL.Array
            The uniformly gridded k-space compressed by the MB factor.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.
          scan_offset : int, 0
            Offset compared to the first acquired scan.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.copy_SMS_adjkspace(
                    self.queue,
                    (sg.shape[0] * sg.shape[1],
                     sg.shape[-2],
                     sg.shape[-1]),
                    None,
                    self._tmp_fft_array.data,
                    s.data,
                    self.shift.data,
                    self.mask.data,
                    np.int32(self.packs),
                    np.int32(self.MB),
                    self.DTYPE_real(self.fft_scale),
                    np.int32(sg.shape[2]/self.packs/self.MB),
                    wait_for=s.events+wait_for))
            for j in range(s.shape[0]):
                self._tmp_fft_array.add_event(
                    self.fft.enqueue_arrays(
                        data=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        result=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        forward=False)[0])

            return (self.prg.copy(self.queue,
                                  (sg.size,),
                                  None,
                                  sg.data,
                                  self._tmp_fft_array.data,
                                  self.DTYPE_real(self.fft_scale),
                                  wait_for=self._tmp_fft_array.events))

        return self.prg.copy_SMS_adj(
                self.queue,
                (sg.shape[0] * sg.shape[1],
                 sg.shape[-2],
                 sg.shape[-1]),
                None,
                sg.data,
                s.data,
                self.shift.data,
                self.mask.data,
                np.int32(self.packs),
                np.int32(self.MB),
                self.DTYPE_real(1),
                np.int32(sg.shape[2]/self.packs/self.MB),
                wait_for=s.events+sg.events+wait_for)

    def FFT(self, s, sg, wait_for=None, scan_offset=0):
        """Perform the forward FFT operation.

        Parameters
        ----------
          s : PyOpenCL.Array
            The uniformly gridded k-space compressed by the MB factor.
          sg : PyOpenCL.Array
            The complex image data.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.
          scan_offset : int, 0
            Offset compared to the first acquired scan.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.copy(
                    self.queue,
                    (sg.size,),
                    None,
                    self._tmp_fft_array.data,
                    sg.data,
                    self.DTYPE_real(1 / self.fft_scale),
                    wait_for=self._tmp_fft_array.events+sg.events+wait_for))

            for j in range(s.shape[0]):
                self._tmp_fft_array.add_event(
                    self.fft.enqueue_arrays(
                        data=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        result=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        forward=True)[0])

            return (
                self.prg.copy_SMS_fwdkspace(
                    self.queue,
                    (s.shape[0] * s.shape[1], s.shape[-2], s.shape[-1]),
                    None,
                    s.data,
                    self._tmp_fft_array.data,
                    self.shift.data,
                    self.mask.data,
                    np.int32(self.packs),
                    np.int32(self.MB),
                    self.DTYPE_real(self.fft_scale),
                    np.int32(sg.shape[2]/self.packs/self.MB),
                    wait_for=s.events+self._tmp_fft_array.events))

        return (
            self.prg.copy_SMS_fwd(
                self.queue,
                (s.shape[0] * s.shape[1], s.shape[-2], s.shape[-1]),
                None,
                s.data,
                sg.data,
                self.shift.data,
                self.mask.data,
                np.int32(self.packs),
                np.int32(self.MB),
                self.DTYPE_real(1),
                np.int32(sg.shape[2]/self.packs/self.MB),
                wait_for=s.events+sg.events+wait_for))


class PyOpenCLRadialNUFFTStreamed(PyOpenCLnuFFT):
    """The streamed version of the non-uniform FFT object.

    This class performs the non-uniform FFT (NUFFT) operation. Linear
    interpolation of a sampled gridding kernel is used to regrid points
    from the non-cartesian grid back on the cartesian grid.

    Parameters
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      kwidth : int
        The width of the sampling kernel for regridding of non-uniform
        kspace samples.
      klength : int
        The length of the kernel lookup table which samples the contineous
        gridding kernel.
      DTYPE : Numpy.dtype
        The comlex precision type. Currently complex64 is used.
      DTYPE_real : Numpy.dtype
        The real precision type. Currently float32 is used.

    Attributes
    ----------
      traj : PyOpenCL.Array
        The comlex sampling trajectory
      dcf : PyOpenCL.Array
        The densitiy compenation function
      ogf (float):
        The overgriddingfactor for non-cartesian k-spaces.
      fft_shape : tuple of ints
        3 dimensional tuple. Dim 0 containts all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale : float32
        The scaling factor to achieve a good adjointness of the forward and
        backward FFT.
      cl_kerneltable (PyOpenCL.Buffer):
        The gridding lookup table as read only Buffer
      cl_deapo (PyOpenCL.Buffer):
        The deapodization lookup table as read only Buffer
      par_fft : int
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft : gpyfft.fft.FFT
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused in each iterations, iterationg over
        all scans to keep the memory footprint low.
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            ctx,
            queue,
            par,
            kwidth=5,
            klength=200,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):

        super().__init__(ctx, queue, par["fft_dim"], DTYPE, DTYPE_real)

        self.ogf = par["N"]/par["dimX"]
        self.fft_shape = (par["NScan"] *
                          par["NC"] *
                          (par["par_slices"] +
                           par["overlap"]),
                          int(par["dimY"]*self.ogf),
                          int(par["dimX"]*self.ogf))
        (kerneltable, kerneltable_FT) = calckbkernel(
            kwidth, self.ogf, par["N"], klength)
        self._kernelpoints = kerneltable.size

        self.fft_scale = DTYPE_real(self.fft_shape[-1])
        self.deapo = 1 / kerneltable_FT.astype(DTYPE_real)
        self._kwidth = kwidth / 2
        self.cl_kerneltable = cl.Buffer(
            self.queue.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=kerneltable.astype(DTYPE_real).data)
        self.cl_deapo = cl.Buffer(
            self.queue.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.deapo.data)
        self.dcf = clarray.to_device(self.queue, par["dcf"])
        self.traj = clarray.to_device(self.queue, par["traj"])
        self._tmp_fft_array = (
            clarray.empty(
                self.queue,
                self.fft_shape,
                dtype=DTYPE))
        self.par_fft = int(self.fft_shape[0] / par["NScan"])
        self.fft = FFT(ctx, queue, self._tmp_fft_array[
            0:self.par_fft, ...],
                       out_array=self._tmp_fft_array[
                           0:self.par_fft, ...],
                       axes=self.fft_dim)

        self._kernelpoints = kerneltable.size
        self._kwidth = kwidth / 2
        self._check = np.ones(par["N"], dtype=DTYPE_real)
        self._check[1::2] = -1
        self._check = clarray.to_device(self.queue, self._check)
        self._gridsize = par["N"]

    def __del__(self):
        """Explicitly delete OpenCL Objets."""
        del self.traj
        del self.dcf
        del self._tmp_fft_array
        del self.cl_kerneltable
        del self.cl_deapo
        del self._check
        del self.queue
        del self.ctx
        del self.prg
        del self.fft

    def FFTH(self, sg, s, wait_for=None):
        """Perform the inverse (adjoint) NUFFT operation.

        Parameters
        ----------
          sg : PyOpenCL.Array
            The complex image data.
          s : PyOpenCL.Array
            The non-uniformly gridded k-space
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        # Zero tmp arrays
        self._tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self._tmp_fft_array.size,
                 ),
                None,
                self._tmp_fft_array.data,
                wait_for=self._tmp_fft_array.events +
                wait_for))
        # Grid k-space
        self._tmp_fft_array.add_event(
            self.prg.grid_lut(
                self.queue,
                (s.shape[0], s.shape[1] * s.shape[2],
                 s.shape[-2] * self._gridsize),
                None,
                self._tmp_fft_array.data,
                s.data,
                self.traj.data,
                np.int32(self._gridsize),
                np.int32(sg.shape[2]),
                self.DTYPE_real(self._kwidth / self._gridsize),
                self.dcf.data,
                self.cl_kerneltable,
                np.int32(self._kernelpoints),
                wait_for=(wait_for + sg.events + s.events +
                          self._tmp_fft_array.events)))
        # FFT
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))
        for j in range(int(self.fft_shape[0] / self.par_fft)):
            self._tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self._tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    result=self._tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    forward=False)[0])
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))
        # Deapodization and Scaling
        return self.prg.deapo_adj(self.queue,
                                  (sg.shape[0] * sg.shape[1] * sg.shape[2],
                                   sg.shape[3], sg.shape[4]),
                                  None,
                                  sg.data,
                                  self._tmp_fft_array.data,
                                  self.cl_deapo,
                                  np.int32(self._tmp_fft_array.shape[-1]),
                                  self.DTYPE_real(self.fft_scale),
                                  self.DTYPE_real(self.ogf),
                                  wait_for=(wait_for + sg.events + s.events +
                                            self._tmp_fft_array.events))

    def FFT(self, s, sg, wait_for=None):
        """Perform the forward NUFFT operation.

        Parameters
        ----------
          s : PyOpenCL.Array
            The non-uniformly gridded k-space.
          sg : PyOpenCL.Array
            The complex image data.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        # Zero tmp arrays
        self._tmp_fft_array.add_event(
            self.prg.zero_tmp(
                self.queue,
                (self._tmp_fft_array.size,
                 ),
                None,
                self._tmp_fft_array.data,
                wait_for=self._tmp_fft_array.events +
                wait_for))
        # Deapodization and Scaling
        self._tmp_fft_array.add_event(
            self.prg.deapo_fwd(
                self.queue,
                (sg.shape[0] * sg.shape[1] * sg.shape[2], sg.shape[3],
                 sg.shape[4]),
                None,
                self._tmp_fft_array.data,
                sg.data,
                self.cl_deapo,
                np.int32(self._tmp_fft_array.shape[-1]),
                self.DTYPE_real(1 / self.fft_scale),
                self.DTYPE_real(self.ogf),
                wait_for=sg.events + self._tmp_fft_array.events))
        # FFT
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))
        for j in range(int(self.fft_shape[0] / self.par_fft)):
            self._tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self._tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    result=self._tmp_fft_array[
                        j * self.par_fft:
                        (j + 1) * self.par_fft, ...],
                    forward=True)[0])
        self._tmp_fft_array.add_event(
            self.prg.fftshift(
                self.queue,
                (self.fft_shape[0],
                 self.fft_shape[1],
                 self.fft_shape[2]),
                None,
                self._tmp_fft_array.data,
                self._check.data))
        # Resample on Spoke

        return self.prg.invgrid_lut(
            self.queue,
            (s.shape[0], s.shape[1] * s.shape[2],
             s.shape[-2] * self._gridsize),
            None,
            s.data,
            self._tmp_fft_array.data,
            self.traj.data,
            np.int32(self._gridsize),
            np.int32(s.shape[2]),
            self.DTYPE_real(self._kwidth / self._gridsize),
            self.dcf.data,
            self.cl_kerneltable,
            np.int32(self._kernelpoints),
            wait_for=(s.events + self._tmp_fft_array.events +
                      sg.events))


class PyOpenCLCartNUFFTStreamed(PyOpenCLnuFFT):
    """The streamed version of the Cartesian FFT object.

    This class performs the FFT operation.

    Parameters
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      DTYPE : Numpy.dtype
        The comlex precision type. Currently complex64 is used.
      DTYPE_real : Numpy.dtype
        The real precision type. Currently float32 is used.

    Attributes
    ----------
      fft_shape : tuple of ints
        3 dimensional tuple. Dim 0 containts all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale : float32
        The scaling factor to achieve a good adjointness of the forward and
        backward FFT.
      par_fft : int
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft : gpyfft.fft.FFT
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused in each iterations, iterationg over
        all scans to keep the memory footprint low.
      mask : PyOpenCL.Array
        The undersampling mask for the Cartesian grid.
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            ctx,
            queue,
            par,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, par["fft_dim"], DTYPE, DTYPE_real)
        self.fft_shape = (par["NScan"] *
                          par["NC"] *
                          (par["par_slices"] +
                           par["overlap"]), par["dimY"], par["dimX"])

        self.par_fft = int(self.fft_shape[0] / par["NScan"])
        if self.fft_dim is not None:
            self.fft_scale = DTYPE_real(
                np.sqrt(np.prod(self.fft_shape[self.fft_dim[0]:])))
            self._tmp_fft_array = (
                clarray.zeros(
                    self.queue,
                    self.fft_shape,
                    dtype=DTYPE))
            self.mask = clarray.to_device(self.queue, par["mask"])
            self.fft = FFT(ctx, queue, self._tmp_fft_array[
                0:self.par_fft, ...],
                           out_array=self._tmp_fft_array[
                               0:self.par_fft, ...],
                           axes=self.fft_dim)

    def __del__(self):
        """Explicitly delete OpenCL Objets."""
        if self.fft_dim is not None:
            del self._tmp_fft_array
            del self.mask
            del self.fft
        del self.queue
        del self.ctx
        del self.prg

    def FFTH(self, sg, s, wait_for=None):
        """Perform the inverse (adjoint) FFT operation.

        Parameters
        ----------
          sg : PyOpenCL.Array
            The complex image data.
          s : PyOpenCL.Array
            The uniformly gridded k-space
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.maskingcpy(
                    self.queue,
                    (self._tmp_fft_array.shape),
                    None,
                    self._tmp_fft_array.data,
                    s.data,
                    self.mask.data,
                    wait_for=s.events+self._tmp_fft_array.events+wait_for))

            for j in range(int(self.fft_shape[0] / self.par_fft)):
                self._tmp_fft_array.add_event(
                    self.fft.enqueue_arrays(
                        data=self._tmp_fft_array[
                            j * self.par_fft:
                            (j + 1) * self.par_fft, ...],
                        result=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        forward=False)[0])
            # Scaling
            return (
                self.prg.copy(
                    self.queue,
                    (sg.size,
                     ),
                    None,
                    sg.data,
                    self._tmp_fft_array.data,
                    self.DTYPE_real(
                        self.fft_scale)))

        return self.prg.copy(
                    self.queue,
                    (s.size,
                     ),
                    None,
                    sg.data,
                    s.data,
                    self.DTYPE_real(1),
                    wait_for=s.events+sg.events+wait_for)

    def FFT(self, s, sg, wait_for=None):
        """Perform the forward FFT operation.

        Parameters
        ----------
          s : PyOpenCL.Array
            The uniformly gridded k-space.
          sg : PyOpenCL.Array
            The complex image data.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.copy(
                    self.queue,
                    (sg.size,
                     ),
                    None,
                    self._tmp_fft_array.data,
                    sg.data,
                    self.DTYPE_real(
                        1 /
                        self.fft_scale), wait_for=wait_for+sg.events))
            for j in range(int(self.fft_shape[0] / self.par_fft)):
                self._tmp_fft_array.add_event(
                    self.fft.enqueue_arrays(
                        data=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        result=self._tmp_fft_array[
                            j * self.par_fft:(j + 1) * self.par_fft, ...],
                        forward=True)[0])

            return (
                self.prg.maskingcpy(
                    self.queue,
                    (self._tmp_fft_array.shape),
                    None,
                    s.data,
                    self._tmp_fft_array.data,
                    self.mask.data,
                    wait_for=s.events+self._tmp_fft_array.events))

        return self.prg.copy(
            self.queue,
            (sg.size,
             ),
            None,
            s.data,
            sg.data,
            self.DTYPE_real(1),
            wait_for=s.events+sg.events+wait_for)


class PyOpenCLSMSNUFFTStreamed(PyOpenCLnuFFT):
    """The streamed version of the Cartesian FFT-SMS object.

    This class performs the FFT operation assuming a SMS acquisition.

    Parameters
    ----------
      ctx : PyOpenCL.Context
        The context for the PyOpenCL computations.
      queue : PyOpenCL.Queue
        The computation Queue for the PyOpenCL kernels.
      par : dict A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      DTYPE : Numpy.dtype
        The comlex precision type. Currently complex64 is used.
      DTYPE_real : Numpy.dtype
        The real precision type. Currently float32 is used.

    Attributes
    ----------
      fft_shape : tuple of ints
        3 dimensional tuple. Dim 0 containts all Scans, Coils and Slices.
        Dim 1 and 2 the overgridded image dimensions.
      fft_scale : float32
        The scaling factor to achieve a good adjointness of the forward and
        backward FFT.
      par_fft : int
        The number of parallel fft calls. Typically it iterates over the
        Scans.
      fft : gpyfft.fft.FFT
        The fft object created from gpyfft (A wrapper for clFFT). The object
        is created only once an reused in each iterations, iterationg over
        all scans to keep the memory footprint low.
      mask : PyOpenCL.Array
        The undersampling mask for the Cartesian grid.
      packs : int
        The distance between the slices
      MB : int
        The multiband factor
      shift : PyOpenCL.Array
        The vector pixel shifts used in the fft computation.
      prg : PyOpenCL.Program
        The PyOpenCL.Program object containing the necessary kernels to
        execute the linear Operator. This will be determined by the
        factory and set after the object is created.
    """

    def __init__(
            self,
            ctx,
            queue,
            par,
            DTYPE=np.complex64,
            DTYPE_real=np.float32):
        super().__init__(ctx, queue, par["fft_dim"], DTYPE, DTYPE_real)
        self.fft_shape = (
            par["NC"] *
            par["NSlice"],
            par["dimY"],
            par["dimX"])

        self.packs = int(par["packs"])
        self.MB = int(par["MB"])
        self.shift = clarray.to_device(
            self.queue, par["shift"].astype(DTYPE_real))

        self.par_fft = int(self.fft_shape[0])

        self.mask = clarray.to_device(self.queue, par["mask"])

        if self.fft_dim is not None:
            self.fft_scale = DTYPE_real(
                np.sqrt(np.prod(self.fft_shape[self.fft_dim[0]:])))
            self._tmp_fft_array = (
                clarray.zeros(
                    self.queue,
                    self.fft_shape,
                    dtype=DTYPE))
            self.fft = FFT(ctx, queue, self._tmp_fft_array[
                0:self.par_fft, ...],
                           out_array=self._tmp_fft_array[
                               0:self.par_fft, ...],
                           axes=self.fft_dim)

    def __del__(self):
        """Explicitly delete OpenCL Objets."""
        if self.fft_dim is not None:
            del self._tmp_fft_array
            del self.fft
        del self.mask
        del self.queue
        del self.ctx
        del self.prg

    def FFTH(self, sg, s, wait_for=None):
        """Perform the inverse (adjoint) FFT operation.

        Parameters
        ----------
          sg : PyOpenCL.Array
            The complex image data.
          s : PyOpenCL.Array
            The uniformly gridded k-space compressed by the MB factor.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.copy_SMS_adjkspace(
                    self.queue,
                    (sg.shape[0] * sg.shape[1],
                     sg.shape[-2],
                     sg.shape[-1]),
                    None,
                    self._tmp_fft_array.data,
                    s.data,
                    self.shift.data,
                    self.mask.data,
                    np.int32(self.packs),
                    np.int32(self.MB),
                    self.DTYPE_real(self.fft_scale),
                    np.int32(sg.shape[2]/self.packs/self.MB),
                    wait_for=s.events+wait_for))
            self._tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self._tmp_fft_array,
                    result=self._tmp_fft_array,
                    forward=False)[0])
            return (self.prg.copy(self.queue,
                                  (sg.size,),
                                  None,
                                  sg.data,
                                  self._tmp_fft_array.data,
                                  self.DTYPE_real(self.fft_scale),
                                  wait_for=self._tmp_fft_array.events))

        return self.prg.copy_SMS_adj(
                self.queue,
                (sg.shape[0] * sg.shape[1],
                 sg.shape[-2],
                 sg.shape[-1]),
                None,
                sg.data,
                s.data,
                self.shift.data,
                self.mask.data,
                np.int32(self.packs),
                np.int32(self.MB),
                self.DTYPE_real(1),
                np.int32(sg.shape[2]/self.packs/self.MB),
                wait_for=s.events+sg.events+wait_for)

    def FFT(self, s, sg, wait_for=None):
        """Perform the forward FFT operation.

        Parameters
        ----------
          s : PyOpenCL.Array
            The uniformly gridded k-space compressed by the MB factor.
          sg : PyOpenCL.Array
            The complex image data.
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
          PyOpenCL.Event: A PyOpenCL event to wait for.
        """
        if wait_for is None:
            wait_for = []
        if self.fft_dim is not None:
            self._tmp_fft_array.add_event(
                self.prg.copy(
                    self.queue,
                    (sg.size,),
                    None,
                    self._tmp_fft_array.data,
                    sg.data,
                    self.DTYPE_real(1 / self.fft_scale),
                    wait_for=self._tmp_fft_array.events+sg.events+wait_for))

            self._tmp_fft_array.add_event(
                self.fft.enqueue_arrays(
                    data=self._tmp_fft_array,
                    result=self._tmp_fft_array,
                    forward=True)[0])
            return (
                self.prg.copy_SMS_fwdkspace(
                    self.queue,
                    (s.shape[0] * s.shape[1], s.shape[-2], s.shape[-1]),
                    None,
                    s.data,
                    self._tmp_fft_array.data,
                    self.shift.data,
                    self.mask.data,
                    np.int32(self.packs),
                    np.int32(self.MB),
                    self.DTYPE_real(self.fft_scale),
                    np.int32(sg.shape[2]/self.packs/self.MB),
                    wait_for=s.events+self._tmp_fft_array.events))

        return (
            self.prg.copy_SMS_fwd(
                self.queue,
                (s.shape[0] * s.shape[1], s.shape[-2], s.shape[-1]),
                None,
                s.data,
                sg.data,
                self.shift.data,
                self.mask.data,
                np.int32(self.packs),
                np.int32(self.MB),
                self.DTYPE_real(1),
                np.int32(sg.shape[2]/self.packs/self.MB),
                wait_for=s.events+sg.events+wait_for))
