## Welcome to ACME Tutorials on Binder! 🚀

This interactive environment allows you to run and modify the notebooks directly in your browser.

### ⚡ Fast Startup - Install Packages As Needed

To keep Binder startup **super fast**, we don't pre-install heavy packages. Instead:

**� Run this at the start of each notebook:**
```python
!pip install healpy astropy astropy-healpix ligo.skymap scipy
```

This installs packages only when you need them!

### �📚 Available Tutorials:

1. **Sky Localization Maps** (`alerts/skymaps.ipynb`)
   - Working with HEALPix sky maps from gravitational wave detections
   - Visualizing probability distributions on the sky
   - Computing credible regions and coverage

2. **Kafka Alert Streaming** (`alerts/kafka_1.ipynb`, `alerts/kafka.ipynb`)
   - Real-time alert processing
   - Connecting to GCN Kafka streams

3. **GWFish Analysis** (`GWFish/gwfish_tutorial.ipynb`)
   - Fisher matrix calculations for gravitational waves
   - Parameter estimation and error analysis

4. **Supernova Signals** (`GWFish/supernova_signals.ipynb`)
   - Analyzing gravitational waves from core-collapse supernovae

5. **Cosmology MCMC** (`GWFish/cosmo_tutorial/cosmo_tutorial.ipynb`)
   - Cosmological parameter estimation
   - MCMC sampling techniques

### 🎯 Quick Start:

1. Navigate to `alerts/` or `GWFish/` in the file browser (left sidebar)
2. Open any `.ipynb` notebook file
3. **Run the package installation cell first** (if present)
4. Then execute the rest of the notebook

### ⚡ Tips:

- **Run cells**: Press `Shift + Enter`
- **Install packages**: Use `!pip install package-name` in a cell
- **Save work**: Use `File > Download` (changes are NOT saved to GitHub)
- **Restart kernel**: `Kernel > Restart` if something goes wrong
- **Timeout**: Sessions expire after ~10 minutes of inactivity

### 📖 Full Documentation:

Visit the main site: https://samueleronchini.github.io/acme_tutorials/

---

**Note**: This is a temporary sandbox environment. Your changes will be lost when the session ends.
