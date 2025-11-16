# ACME Tutorials

[![Deploy Jupyter Book](https://github.com/samueleronchini/acme_tutorials/actions/workflows/deploy.yml/badge.svg)](https://github.com/samueleronchini/acme_tutorials/actions/workflows/deploy.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/samueleronchini/acme_tutorials/main)

Interactive tutorials for gravitational wave astronomy and related topics, powered by Jupyter Book and Binder.

🌐 **Live Site**: https://samueleronchini.github.io/acme_tutorials/

🚀 **Interactive Notebooks**: Click the rocket icon on any tutorial page to launch with Binder!

## Contents

- **Sky Maps Tutorial**: Working with HEALPix sky maps from gravitational wave detections
- **Kafka Listener**: Real-time alert streaming and processing
- **GWFish Tutorial**: Gravitational wave Fisher matrix analysis
- **Supernova Signals**: Analyzing supernova gravitational wave signals
- **Cosmology Tutorial**: Cosmological parameter estimation with MCMC

## Setup Instructions

### 1. Enable GitHub Pages

To deploy your site, you need to enable GitHub Pages:

1. Go to your repository settings: https://github.com/samueleronchini/acme_tutorials/settings/pages
2. Under "Build and deployment":
   - **Source**: Select "GitHub Actions"
3. Save the changes

### 2. Push Changes and Deploy

Once you've pushed these configuration files to your repository, the GitHub Actions workflow will automatically:
- Build the Jupyter Book
- Deploy it to GitHub Pages
- Make it available at: https://samueleronchini.github.io/acme_tutorials/

### 3. Run Locally

To build and preview the book locally:

```bash
# Clone the repo
git clone https://github.com/samueleronchini/acme_tutorials.git
cd acme_tutorials

# Install dependencies
pip install "jupyter-book<1.0"
pip install -r requirements.txt

# Build and view the site
jupyter-book build .
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

## Binder Integration

Binder is automatically configured for all notebooks. When users click the 🚀 rocket icon on any page:
- Binder will create a temporary environment with all dependencies from `requirements.txt`
- Users can run and modify the notebooks interactively
- Changes are not saved back to the repository (sandbox environment)

## File Structure

```
acme_tutorials/
├── _config.yml           # Jupyter Book configuration
├── _toc.yml             # Table of contents
├── intro.md             # Homepage
├── requirements.txt     # Python dependencies
├── .github/
│   └── workflows/
│       └── deploy.yml   # GitHub Actions workflow
├── alerts/              # Alert processing tutorials
│   └── skymaps.ipynb
│   └── kafka_listener.ipynb
└── GWFish/             # GWFish tutorials
    └── gwfish_tutorial.ipynb
    └── supernova_signals.ipynb
    └── cosmo_tutorial/
        └── cosmo_tutorial.ipynb
```

## Adding New Notebooks

1. Add your notebook file to the appropriate directory
2. Update `_toc.yml` to include the new notebook:
   ```yaml
   - file: path/to/notebook
     title: Notebook Title
   ```
3. Push changes - the site will rebuild automatically

## Customization

### Theme and Appearance
Edit `_config.yml` to customize:
- Site title and author
- Theme options
- Launch buttons
- Repository links

### Dependencies
Update `requirements.txt` with any new packages needed by your notebooks.

### Content Structure
Modify `_toc.yml` to reorganize chapters and sections.

## Troubleshooting

### Build Fails
- Check the Actions tab for error logs
- Ensure all notebook paths in `_toc.yml` are correct
- Verify all dependencies are in `requirements.txt`

### Binder Times Out
- Large notebooks may take time to load
- Consider reducing notebook complexity or splitting into smaller notebooks
- Increase timeout in `_config.yml` under `execute.timeout`

## Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [Binder Documentation](https://mybinder.readthedocs.io/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)

## License

[Add your license information here]

## Contact

Samuele Ronchini - [Add contact information]
