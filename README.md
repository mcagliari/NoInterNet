# No-Interloper Network - NoInterNet

[![arXiv](https://img.shields.io/badge/arXiv-2204.13713-b31b1b.svg)](https://arxiv.org/abs/2504.06919)
[![Data Docs](https://img.shields.io/badge/Data%20docs-available-blue.svg)](https://quijote-simulations.readthedocs.io/en/latest/interlopers.html)

This code was used for the analysis presented in [Cagliari et al. (2025)](https://arxiv.org/abs/2504.06919), which explores the correction of line interloper contamination in measured summary statistics using machine learning, focusing on the power spectrum monopole. We generated contaminated catalogs with varying interloper fractions, using [Quijote](https://quijote-simulations.readthedocs.io/en/latest/index.html)'s Friend-of-Friend catalogs with a $512^3$ resolution for $1000$ snapshots in the fiducial cosmology, as well as for the $\Lambda$CDM Latin hypercube and the Big Sobol Sequence. We simulated two types of line interlopers:  

- **"Inbox" interlopers**, which are highly correlated with the target sample (displacing halos within the same snapshot).  
- **"Outbox" interlopers**, which have low correlation with the target sample (displacing halos from a different snapshot).  

The method achieves high accuracy ($<1\%$ error) at fixed cosmology, and while performance degrades when cosmological parameters vary, including bispectrum information significantly mitigates this. All contaminated simulations are made publicly available [here](https://quijote-simulations.readthedocs.io/en/latest/interlopers.html).

# Requirements

The libaries required to build the contaminated catalogues and measure the statistics are:

- `numpy`
- `[Pylians3](https://pylians3.readthedocs.io/en/master/)`
- `[pySpectrum](https://github.com/changhoonhahn/pySpectrum)`

The library to train the model and reproduce the plots are:

- `matplotlib`
- `pytorch`
- `optuna`

# Acknowledgements

This work has been done thanks to the facilities offered by the Univ. Savoie Mont Blanc - CNRS/IN2P3 MUST computing center.

# Team

- Marina Silvia Cagliari (LAPTh, France)
- Azadeh Moradinezhad (LAPTh, France)
- Francisco Villaescusa-Navarro (Simons/Princeton, USA)

# Citation

If you use this code, please link this repository and cite [Cagliari et al. (2025)](https://arxiv.org/abs/2504.06919). 