from setuptools import setup

setup(name='mbpq',
      version='0.1',
      description='Model-based parameter quantification on the GPU',
      url='https://github.com/MaierOli2010/MBPQ',
      author='Oliver Maier',
      author_email='oliver.maier@tugraz.at',
      license='Apache-2.0',
      packages=setuptools.find_packages(),
      install_requires=[
        'pyopencl',
        'numpy',
        'h5py'
        'ipyparallel',
        'mako',
        'matplotlib'],
      zip_safe=False) 
