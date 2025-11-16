#!/usr/bin/env python3
"""
Test script to verify all required packages for ACME tutorials
Run this to check if your environment is properly configured
"""

import sys

def test_import(package_name, import_name=None):
    """Try to import a package and report status"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown version')
        print(f"✅ {package_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} NOT FOUND - {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name:20s} Import error: {str(e)}")
        return False

def main():
    print("="*60)
    print("ACME Tutorials - Package Verification")
    print("="*60)
    print()
    
    # Core scientific packages
    print("📊 Core Scientific Packages:")
    core_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
    ]
    
    core_ok = all(test_import(name, imp) for name, imp in core_packages)
    print()
    
    # Astronomy packages
    print("🌌 Astronomy Packages:")
    astro_packages = [
        ('astropy', 'astropy'),
        ('astropy-healpix', 'astropy_healpix'),
        ('healpy', 'healpy'),
        ('ligo.skymap', 'ligo.skymap'),
    ]
    
    astro_ok = all(test_import(name, imp) for name, imp in astro_packages)
    print()
    
    # Data handling
    print("💾 Data Handling:")
    data_packages = [
        ('h5py', 'h5py'),
        ('lxml', 'lxml'),
        ('tables', 'tables'),
    ]
    
    data_ok = all(test_import(name, imp) for name, imp in data_packages)
    print()
    
    # Jupyter
    print("📓 Jupyter Ecosystem:")
    jupyter_packages = [
        ('jupyterlab', 'jupyterlab'),
        ('ipykernel', 'ipykernel'),
        ('ipywidgets', 'ipywidgets'),
    ]
    
    jupyter_ok = all(test_import(name, imp) for name, imp in jupyter_packages)
    print()
    
    # Utilities
    print("🔧 Utilities:")
    util_packages = [
        ('xmltodict', 'xmltodict'),
        ('corner', 'corner'),
        ('tqdm', 'tqdm'),
        ('sympy', 'sympy'),
    ]
    
    util_ok = all(test_import(name, imp) for name, imp in util_packages)
    print()
    
    # Tutorial-specific
    print("🎓 Tutorial-Specific Packages:")
    tutorial_packages = [
        ('GWFish', 'GWFish'),
    ]
    
    tutorial_ok = all(test_import(name, imp) for name, imp in tutorial_packages)
    print()
    
    # Optional packages (not on Binder)
    print("⚡ Optional Packages (not available on Binder):")
    optional_packages = [
        ('lalsuite', 'lal'),
        ('confluent-kafka', 'confluent_kafka'),
        ('gcn-kafka', 'gcn_kafka'),
    ]
    
    for name, imp in optional_packages:
        test_import(name, imp)
    print()
    
    # Summary
    print("="*60)
    print("📋 Summary:")
    print("="*60)
    
    all_required_ok = core_ok and astro_ok and data_ok and jupyter_ok and util_ok and tutorial_ok
    
    if all_required_ok:
        print("✅ All REQUIRED packages are installed!")
        print("🚀 Your environment is ready for ACME tutorials")
        return 0
    else:
        print("❌ Some REQUIRED packages are missing")
        print("🔧 Install missing packages using:")
        print("   conda env create -f binder/environment.yml")
        print("   OR")
        print("   pip install -r binder/requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
