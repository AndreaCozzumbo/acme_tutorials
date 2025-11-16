# Alerts Tutorial Files

This directory contains tutorial files for gravitational wave alert handling and sky localization.

## Large Data Files

### `bayestar.fits` (96 MB)
This is a LIGO/Virgo sky localization map used in the `skymaps.ipynb` tutorial. 

**Note for Binder users:** 
- If this file is missing from the repository (due to Git LFS or size limits), the `postBuild` script will automatically download it when launching Binder
- Alternatively, you can download it from LIGO/Virgo public data archives

### `bayestar.multiorder.fits,1` (759 KB)
Multi-order HEALPix version of the same skymap, already included in the repository.

## Environment Setup

All required Python packages are pre-installed via `environment.yml`:
- `healpy` - HEALPix map manipulation
- `ligo.skymap` - LIGO/Virgo sky localization tools
- `astropy`, `astropy-healpix` - Astronomy utilities
- `scipy`, `matplotlib` - Scientific computing and visualization

**No manual installation needed** when using Binder or conda environment!
