---
title: 'PyQMRI: Accelerated Quantitative MRI toolbox'
tags:
  - Python
  - MRI
  - PyOpenCL
  - quantitative imaging
  - parameter mapping
authors:
  - name: Oliver Maier^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0002-7800-0022
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Rudolf Stollberger
    orcid: 0000-0002-4969-3878
    affiliation: "1, 2"
affiliations:
 - name: Institute of Medical Engineering, Graz University of Technology, Graz, Austria
   index: 1
 - name: BioTechMed Graz
   index: 2
date: 15 September 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Quantitative MRI (qMRI) aims at identifying the underlying physical tissue quantities 
that define the contrast in an imaging experiment. Under certain simplifications,
analytical expressions are available to describe the relation between image
intensity and physical parameters of tissue. Using several measurements with 
varying sequence parameters it is possible to solve the associated inverse problem.

The increased measurement time of such experiments is typically tackeld by 
undersampling the data acquisiton. However, the reduced amount of data as well 
as the typical non-linear problem structer require dedicated numerical solution strategies
which lead to prolonged reconstruction times. An effect that gets even worse in
volumetric imaging. 


# Statement of need 

'PyQMRI' aims at reducing the required reconstruction time by means of a
highly parallelized PyOpenCL implementation of an state-of-the-art fitting algorithm 
while maintaining the easy-to-use properties of a Python package.
In addition to fitting small date completely on the GPU an efficient
double-buffering based solution strategy is implemented. Double-buffering 
allows to overlap computation and memory transfer from/to the GPU, thus
hiding the associated memory latency. By overlapping the transfered blocks
it is possible to pass on 3D information utilizing finite differences based
regularization strategies. 

Currently 3D acuqisitions with at least one fully sampled dimension can
be reconstructed on the GPU, including stack-of-X acquisitions or 3D Cartesian
based imaging. Of course 2D data can be reconstructed as well. Fitting is based
on an iterativly regularized Gauss-Newton (IRGN) approach combined with 
a primal-dual inner loop. Regulariaztion strategies include total variation (TV)
and total generalized variation (TGV) using finite differences gradient operations.

'PyQMRI' comes with several pre-implemented quantiative models. In addition
new models can be introduced via a simple text file, utilizing the power
of 'SymPy'. Fitting can be initiated via a CLI or by importing the package
into a Python script. 

'PyQMRI' and its precedors have been succesfully used in several scientific
publications. To the best of the authors knowledge 'PyQMRI'
is the only availabel Python toolbox that offers real 3D regularization 
for arbitrary large volumetric data while simultaneously utilizing the computation
power of recent GPUs.

# Algorithmic
The general problem structure deald with in 'PyQMRI' is as follows:

$$
\underset{u,v}{\min}\quad 
\frac{1}{2}\sum_{n=1}^{N_d}\|A_{\phi,t_n}(u)-d_n\|_2^2 
+\nonumber \gamma( \alpha_0\|\nabla u - v\|_{1,2,F} + 
\alpha_1\|\mathcal{E}v\|_{1,2,F})
$$

which includes a non-linear forward operator ($A$) and non-smooth regularization due to 
the $L^1$-norms of the T(G)V functional. For $\alpha_1=0$ and $v=0$ the problem
amounts to simple TV regularization. To further improve the quality of the 
reconstructed parameter maps 'PyQMRI' uses a Frobenius norm to join spatial
information from all maps in the T(G)V functionals.

# Acknowledgements

Oliver Maier acknowledges grant support from the Austrian Academy of Sciences under award DOC-Fellowship 24966.

# References
