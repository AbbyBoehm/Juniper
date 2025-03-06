import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from juniper.stage5.bin_light_curves import time_bin

def plot_spectrum(waves,depths,errors,bin_f,wave_bounds,spec_type):
    """Plots the given spectrum type.

    Args:
        waves (np.array): wavelengths for each depth.
        depths (np.array): depth of whatever prescription was chosen.
        errors (np.array): uncertainties on the depths.
        bin_f (float): factor to bin the spectrum down by.
        wave_bounds (lst): boundaries for wavelengths.
        If not None, clip spectrum to these wavelengths.
        spec_type (str): type of spectrum.

    Returns:
        fig, ax: matplotlib plot of spectrum.
    """
    # Copy everything for safety.
    w, d, e = np.copy(waves), np.copy(depths), np.copy(errors)

    # Check if user wants these binned.
    if bin_f != 1:
        w = time_bin(w,bin_f,'mean')
        d = time_bin(d,bin_f,'mean')
        e = time_bin(e,bin_f,'quadrature')

    # Check if user wants to trim wavelengths.
    if wave_bounds:
        # Cut the plot off at where the waves are in these bounds.
        delete_these = []
        for i, wi in enumerate(w):
            if wi < wave_bounds[0] or wi > wave_bounds[1]:
                delete_these.append(i)
        w = np.delete(w,delete_these)
        d = np.delete(d,delete_these)
        e = np.delete(e,delete_these)
        
    fig, ax = plt.subplots(figure=(10,7))
    ax.errorbar(w,d,yerr=e,fmt='ko',ls='none',capsize=3)
    ax.set_xlabel(r"wavelength [$\mu$m]")
    ax.set_ylabel('depth [{}]'.format(spec_type))
    if spec_type == 'fpfs':
        ax.axhline(y=0,ls='--',color='k')

    return fig, ax
