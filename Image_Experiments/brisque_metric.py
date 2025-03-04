import subprocess
import sys

def install_package(package):
    """Function to install a Python package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Example usage
install_package("image-quality")  # This will install the numpy package

from imquality import brisque


def compute_brisque(img_ori, img_mod)
    return brisque.score(img_mod)
