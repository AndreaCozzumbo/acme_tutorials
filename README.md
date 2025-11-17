# ACME Tutorials

[![Deploy Jupyter Book](https://github.com/samueleronchini/acme_tutorials/actions/workflows/deploy.yml/badge.svg)](https://github.com/samueleronchini/acme_tutorials/actions/workflows/deploy.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/samueleronchini/acme_tutorials/main)

<!-- markdownlint-disable MD033 -->
<p align="center">
   <img src="alerts/images/acme.png" alt="ACME Tutorials logo" style="width:75%;max-width:720px;border-radius:18px;box-shadow:0 20px 45px rgba(15,23,42,0.18);">
</p>
<!-- markdownlint-enable MD033 -->

Interactive tutorials for gravitational wave astronomy and related topics, powered by Jupyter Book and Binder.

🌐 **Live Site**: <https://samueleronchini.github.io/acme_tutorials/>

🚀 **Interactive Notebooks**: Click the rocket icon on any tutorial page to launch with Binder!


## Setup Instructions

### 1. Enable GitHub Pages

To deploy your site, you need to enable GitHub Pages:

1. Go to your repository settings: <https://github.com/samueleronchini/acme_tutorials/settings/pages>
2. Under "Build and deployment":
   - **Source**: Select "GitHub Actions"
3. Save the changes

### 2. Push Changes and Deploy

Once you've pushed these configuration files to your repository, the GitHub Actions workflow will automatically:

- Build the Jupyter Book
- Deploy it to GitHub Pages
- Make it available at: <https://samueleronchini.github.io/acme_tutorials/>

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

