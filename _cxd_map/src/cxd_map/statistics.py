import numpy as np
from scipy.ndimage import center_of_mass as scipy_com


def center_of_mass(imgs: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass for a 3D image.
    """

    return np.array(scipy_com(imgs))
