Running the reconstruction
==========================
.. role:: bash(code)
   :language: bash
   
.. role:: python(code)
   :language: python
   
First, start an ipcluster for speeding up the coil sensitivity estimation:

:bash:`ipcluster start -n N`

where N amounts to the number of processe to be used. If -n N is ommited, 
as many processes as number of CPU cores available are started.

Reconstruction of the parameter maps can be started either using the terminal by typing:

:bash:`pyqmri`

or from python by:

.. code-block:: python

          import pyqmri
          pyqmri.run()

A list of accepted flags can be printed using 

:bash:`pyqmri -h`

or by fewing the documentation of pyqmri.pyqmri in python.

If reconstructing fewer slices from the volume than acquired, slices will be picked symmetrically from the center of the volume. E.g. reconstructing only a single slice will reconstruct the center slice of the volume. 
