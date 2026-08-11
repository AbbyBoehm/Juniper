import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def decorrelate(phot_tseries, xtrend, ytrend, fwhmtrend, t, inpt_dict):
    """Decorrelates photometric time-series and uncertainties using x, y, and FWHM.

    Args:
        phot_tseries (np.array): extracted photometric time-series.
        xtrend (np.array): the trend in the PSF's x position.
        ytrend (np.array): the trend in the PSF's y position.
        fwhmtrend (np.array): the trend in the PSF's full width at half maximum.
        t (np.array): timestamps for each photometric point, for plotting.
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

    # Mask transit/eclipse so it does not drive fit, if asked.
    tstart, tend = None, None
    data_mask = np.zeros_like(phot_tseries)
    if inpt_dict["omit_idxs"]:
        idx_start, idx_end = inpt_dict["omit_idxs"]
        data_mask[idx_start:idx_end] = 1
        tstart, tend = t[idx_start], t[idx_end]

    # Define residuals function.
    def _residuals(theta,flux,x,y,fwhm,data_mask,inpt_dict):
        # Use the inpt_dict to unpack theta.
        n_coeff_x = inpt_dict["x_order"]+1
        n_coeff_y = inpt_dict["y_order"]+1
        coeff_x = theta[:n_coeff_x]
        coeff_y = theta[n_coeff_x:n_coeff_x+n_coeff_y]
        coeff_f = theta[n_coeff_x+n_coeff_y:]

        # polyval each trend.
        trend = 1 + (np.polynomial.polynomial.polyval(x,coeff_x)      \
                * np.polynomial.polynomial.polyval(y,coeff_y)    \
                * np.polynomial.polynomial.polyval(fwhm,coeff_f))
        
        # Take residuals.
        res = (flux-trend)**2

        # Apply mask to res.
        res = np.ma.masked_array(res,mask=data_mask)

        return np.ma.sum(res)
    
    # Initialize x0 as all 100 ppm.
    n_coeff_x = inpt_dict["x_order"]+1
    n_coeff_y = inpt_dict["y_order"]+1
    n_coeff_f = inpt_dict["fwhm_order"]+1
    x0 = [100e-6 for x in range(n_coeff_x+n_coeff_y+n_coeff_f)]
    
    # Initiate quick least-squares fit.
    result = minimize(_residuals,x0=x0,
                      args=(phot_tseries/np.nanmedian(phot_tseries),xtrend,ytrend,fwhmtrend,
                            data_mask,inpt_dict))
    
    # Unpack result and generate model.
    xf = result.x
    coeff_x = xf[:n_coeff_x]
    coeff_y = xf[n_coeff_x:n_coeff_x+n_coeff_y]
    coeff_f = xf[n_coeff_x+n_coeff_y:]

    # polyval each trend.
    trend_x = np.polynomial.polynomial.polyval(xtrend,coeff_x)
    trend_y = np.polynomial.polynomial.polyval(ytrend,coeff_y)
    trend_f = np.polynomial.polynomial.polyval(fwhmtrend,coeff_f)

    # Create decorrelated time-series.
    decorr_tseries = phot_tseries/(1+(trend_x*trend_y*trend_f))
    
    if (plot_ints or save_ints):
        # Plot all three trends and the detrended data.
        fig,ax = plt.subplots(nrows=5,figsize = (10,10),sharex=True)
        plt.subplots_adjust(hspace=0)
        ax[0].plot(t,trend_x,color='red',marker='o',ls='--',label='X-position')
        ax[1].plot(t,trend_y,color='green',marker='o',ls='--',label='Y-position')
        ax[2].plot(t,trend_f,color='blue',marker='o',ls='--',label='Full Width at Half-Maximum')
        ax[3].plot(t,1+(trend_x*trend_y*trend_f),color='indigo',marker='o',ls='--',label='Total Trend')
        ax[4].plot(t,decorr_tseries,color='k',marker='o',ls='--',label='Decorrelated Time-Series')
        ax[4].set_xlabel('Exposure Time [MJD]')
        for k in range(5):
            ax[k].set_ylabel('Trend [a.u.]')
            ax[k].legend(loc='upper right')
        ax[4].set_ylabel('Flux [DN/s]')

        if None not in (tstart, tend):
            for k in range(5):
                for ti in (tstart, tend):
                    ax[k].axvline(x=ti,color='grey',ls=':',alpha=0.5)
        
        if save_ints:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_decorrelation.png'),
                        dpi=300, bbox_inches='tight')
        if plot_ints:
            plt.show(block=True)
        plt.close()

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return decorr_tseries

