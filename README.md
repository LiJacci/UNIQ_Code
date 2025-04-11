# UNIQ: Unsupervised Point Cloud Quality Assessment Via Natural Statistics Modeling
This repository contains the source code of UNIQ.

For **NR mode**:

  use: `python pristine.py` to build the pristine parameters. 
  
  Then: `python compute_quality.py` to evaluate distorted point clouds.
  
  We give the k-fold pristine model in the folder 'perdataset_pristine'. Please read 'README.txt' in the folder to use them.

For **RR mode**:

  use: `python reference.py` to build the reference parameters.
  
  Then: `python test_quality.py` to evaluate distorted point clouds.
  
