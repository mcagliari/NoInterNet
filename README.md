# No-Interloper Network - NoInterNet - Line Intensity Mapping

[![arXiv](https://img.shields.io/badge/arXiv-newnumber-b31b1b.svg)](new link)

This code was used for the analysis presented in [Cagliari et al. (2025)](new link), which explores the correction of line interloper contamination in line intensity mapping surveys using machine learning. We develop a neural network to separate target, interloper, and continuum power spectra incorporating two wastrophysical uncertainties. We find accurate recovery of the target ($≤3\%$) and continuum spectra, though interlopers remain challenging.

# Data

The data used for this will be made public in the future.

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
The datasets for this project were produced with computing and storage resources by GENCI at IDRIS, on the CSL partition of the supercomputer Jean Zay.

# Team

- Marina Silvia Cagliari (LAPTh, France)
- Zucheng Gao (LAPTh, France)
- Azadeh Moradinezhad (LAPTh, France)

# Citation

If you use this code and trained networks, please link this repository and cite [Cagliari et al. (2025)](new link). 