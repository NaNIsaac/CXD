from scipy.ndimage import center_of_mass
import numpy as np


def get_com(images: np.ndarray) -> np.ndarray:
    return np.array(center_of_mass(images))
