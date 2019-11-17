# The-No-U-Turn-Sampler-Adaptively-Setting-Path-Lengths-in-Hamiltonian-Monte-Carlo

This repository implements code for NUTS and other methods that adaptively set path lengths in Hamiltonian Monte Carlo. In particular, I implemented the methods of:

- Hoffman and Gelman (2014): The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo https://arxiv.org/abs/1111.4246 
- Wu et al. (2019): Faster Hamiltonian Monte Carlo by Learning Leapfrog Scale https://arxiv.org/abs/1810.04449

For the implementation of NUTS I borrowed the code from: https://github.com/trevorcampbell/bayesian-coresets/blob/master/examples/hilbert_logistic_poisson_regression/inference.py
