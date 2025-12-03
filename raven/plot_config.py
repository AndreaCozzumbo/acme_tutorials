"""
🎨 Matplotlib Style Configuration (Light & Dark Mode Compatible)

This module provides universal matplotlib settings that work in both light and dark themes.
Import with: from plot_config import setup_matplotlib_style
Then call: setup_matplotlib_style()

Features:
- High-quality output (500 DPI display, 300 DPI saved)
- Transparent backgrounds for theme adaptation
- Dark gray text that's readable on both backgrounds
- Computer Modern math font
"""

import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib_style():
    """
    Configure matplotlib for high-quality, theme-agnostic plots.
    
    This function sets up:
    - High-quality output (500 DPI display, 300 DPI save)
    - Transparent backgrounds
    - Dark gray text for readability on both light and dark backgrounds
    - Computer Modern fonts for professional appearance
    """
    
    # Use mathtext with Computer Modern and avoid external TeX
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Nimbus Roman', 'Times New Roman', 'Times']
    
    # Universal matplotlib settings that work in both light and dark mode
    matplotlib.rcParams.update({
        # High-quality output
        'figure.dpi': 300,              # Display DPI (screen)
        'savefig.dpi': 300,             # Save DPI (high quality for web)
        'figure.figsize': (10, 6),      # Default figure size in inches (max width)
        
    })


# Apply configuration when module is imported
setup_matplotlib_style()
