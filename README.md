# GAN-BH
Generative Modeling for Black Hole Astrophysics

# Copyright
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so (Copyright request O4927).

# Description
GAN-BH is written in python with pytorch library. It contains several different generative modeling architectures: simple GAN, conditional GAN, Wasserstein GAN, and diffusion model. It also contains physics-informed model including critic distance computation and correlation computation to all of those architecture. It has GPU capabilities to perform the work faster. 

## Dependencies

Pytorch (include torch vision), Numpy, Matplotlib


## Usage

We expect that user have appropriate training data e.g. black hole light curve in numpy array `npy`. Currently, different generative models are implemented in 1D and 2D. 

The code has two different modes

To train

`python lc_gan_1d.py --mode train --epochs 50 --data_dir <your data location>`

To generate image

`python lc_gan_1d.py --mode generate --lambda_0 25.0`
