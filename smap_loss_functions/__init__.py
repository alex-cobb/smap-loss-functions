"""SMAP loss functions after Koster et al (2017), doi:10.1175/jhm-d-16-0285.1"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Backport for Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('smap_loss_functions')
except PackageNotFoundError:
    # Package not installed; set default version
    __version__ = '0.0.0'
