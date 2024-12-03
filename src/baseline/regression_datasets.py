"""
Register the regression datasets that should be used with the baselines. They need to be registered with
the OpenDataVal benchmark to be accessible by the benchmark.

The notebook `select_regression_datasets_from_CTR23.ipynb` is used to generate this code.
"""

from opendataval.dataloader import Register
from opendataval.dataloader.datasets.datasets import load_openml


@Register("kin8nm")
def download_kin8nm():
    """Regression data set registered as ``"kin8nm"``."""
    return load_openml(data_id=44980, is_classification=False)


@Register("white_wine")
def download_white_wine():
    """Regression data set registered as ``"white_wine"``."""
    return load_openml(data_id=44971, is_classification=False)


@Register("cpu_activity")
def download_cpu_activity():
    """Regression data set registered as ``"cpu_activity"``."""
    return load_openml(data_id=44978, is_classification=False)


@Register("pumadyn32nh")
def download_pumadyn32nh():
    """Regression data set registered as ``"pumadyn32nh"``."""
    return load_openml(data_id=44981, is_classification=False)


@Register("superconductivity")
def download_superconductivity():
    """Regression data set registered as ``"superconductivity"``."""
    return load_openml(data_id=44964, is_classification=False)
