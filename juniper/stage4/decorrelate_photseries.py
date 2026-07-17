import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def decorrelate(phot_tseries, xtrend, ytrend, fwhmtrend, inpt_dict):
    """Decorrelates photometric time-series and uncertainties using x, y, and FWHM.

    Args:
        phot_tseries (np.array): extracted photometric time-series.
        xtrend (np.array): the trend in the PSF's x position.
        ytrend (np.array): the trend in the PSF's y position.
        fwhmtrend (np.array): the trend in the PSF's full width at half maximum.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array, np.array: aligned spectra and errors.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Decorrelating extracted photometric time-series...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Initialize decorrelated array.
    decorr_tseries = []

    # Mask transit/eclipse so it does not drive fit.
    data_mask = np.zeros_like(phot_tseries)
    if inpt_dict["omit_idxs"]:
        idx_start, idx_end = inpt_dict["omit_idxs"]
        data_mask[idx_start:idx_end] = 1
    fit_tseries = np.ma.masked_array(phot_tseries,mask=data_mask)

    # Define detrending models by order.
    def x_detrend(trend,order=inpt_dict["x_order"]):
        # Set up 
        poly_coeffs = [1.0]
        poly_coeffs.extend(coeffs)
    
        # Then evaluate and return with numpy.
        return np.polynomial.polynomial.polyval(xpos,poly_coeffs)
    
    def _residuals():
        return
    
    # Initiate quick least-squares fit.
    result = minimize(_residuals,x0=x0,
                      args=(phot_tseries,xtrend,ytrend,fwhmtrend))
        
    if (plot_ints or save_ints):
        # Plot a stack of each trend model.
        fig,ax = plt.subplots(figsize = (10,5))
        ax.plot(cpix, oneD_spec[0,:], color='teal', alpha=0.5,label='pre-shifted')
        ax.plot(shift_cpix, align_spec[0], color='red',alpha=0.75,label='shifted')
        for i in range(1,oneD_spec.shape[0]):
            ax.plot(cpix, oneD_spec[i,:], color='teal', alpha=0.5)
            ax.plot(shift_cpix, align_spec[i], color='red',alpha=0.75)
        ax.set_xlabel('Cross-dispersion Position [Pixels]')
        ax.set_ylabel('Flux [DN]')
        ax.legend(loc='upper right')
        ax.set_title('Shifted spectra')
        if save_ints:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_shifted_spectra.png'),
                        dpi=300, bbox_inches='tight')
        if plot_ints:
            plt.show(block=True)
        plt.close()

    align_spec = np.array(align_spec)
    align_err = np.array(align_err)

    if (plot_step or save_step):
        # Plot the detrending phot series.
        plt.scatter(oneD_time, shifts, color='midnightblue')
        plt.xlabel("Exposure Time [BJD TDB]")
        plt.ylabel("Shift [Pixels]")
        plt.title('Cross-correlation shifts')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_spectral_shifts.png'),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    return align_spec, align_err, np.array(shifts)

