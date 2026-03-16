import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def align(oneD_spec, oneD_err, wav_sols, oneD_time, inpt_dict):
    """Aligns 1D spectra and uncertainties using cross-correlation.

    Args:
        oneD_spec (np.array): extracted 1D spectra.
        oneD_err (np.array): extracted 1D errors.
        wav_sols (np.array): wavelength solutions for each spectrum.
        oneD_time (np.array): timestamps for each spectrum.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array, np.array: aligned spectra and errors.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Aligning extracted 1D spectra...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Initialize aligned arrays and track measured shifts.
    align_spec, align_err, align_wav = [], [], []
    shifts = []

    # Define template spectrum as the median in time and get fit parameters.
    med_spec = np.median(oneD_spec, axis=0)
    cpix = np.arange(med_spec.shape[0])
    tspc = inpt_dict["trim_spec"]
    hrf = inpt_dict["high_res_factor"]
    tfit = inpt_dict["trim_fit"]
    
    # Measure the shift with each spectrum.
    for i in tqdm(range(oneD_spec.shape[0]),
                  desc='Measuring displacements through cross-correlation...',
                  disable=(not time_ints)):
        # Cross-correlate the spectra with the template.
        shift = cross_correlate(oneD_spec[i,:],med_spec,
                                tspc=tspc,hrf=hrf,tfit=tfit)
        shifts.append(shift)
    
    # Then, align.
    for i in tqdm(range(oneD_spec.shape[0]),
                  desc='Aligning spectra...',
                  disable=(not time_ints)):
        shift_cpix = cpix + shifts[i]
        interp_spec = interp1d(cpix, oneD_spec[i,:], kind='linear', fill_value='extrapolate')
        align_spec.append(interp_spec(shift_cpix))

        interp_err = interp1d(cpix, oneD_err[i,:], kind='linear', fill_value='extrapolate')
        align_err.append(interp_err(shift_cpix))
        
    if (plot_ints or save_ints):
        # Plot a stack of all raw and shifted specs.
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

def cross_correlate(spec, template, tspc, hrf, tfit):
    """Cross-correlates the spectrum with the template to find the shift.
    Adapted from ExoTiC-JEDI align_spectra method.

    Args:
        spec (np.array): a 1D spectrum to be cross-correlated.
        template (np.array): the template 1D spectrum to measure shifts.
        tspc (int): how many points to trim off the ends of the spectrum.
        hrf (float): factor for interpolating to higher res.
        tfit (int): how many points to trim off the ends of the fit.

    Returns:
        float: the shift in the dispersion direction.
    """
    # Trim spec for getting the shift.
    x = np.copy(spec[tspc:-tspc])
    y = np.copy(template)

    # Eliminate nan.
    x = np.where(np.isnan(x),np.nanmedian(x),x)
    y = np.where(np.isnan(y),np.nanmedian(y),y)
    
    # Interpolate to higher res.
    intrp_fx = interp1d(np.arange(0,x.shape[0]),x,kind="cubic")
    x_hr = intrp_fx(np.arange(0,x.shape[0]-1,hrf))
    intrp_fy = interp1d(np.arange(0,y.shape[0]),y,kind="cubic")
    y_hr = intrp_fy(np.arange(0,y.shape[0]-1,hrf))
    
    # Level.
    x_hr -= np.linspace(x_hr[0],x_hr[-1],np.shape(x_hr)[0])
    y_hr -= np.linspace(y_hr[0],y_hr[-1],np.shape(y_hr)[0])
    
    # Cross-correlate.
    correlation = signal.correlate(x_hr,y_hr,mode="full")
    lags = signal.correlation_lags(x_hr.size,y_hr.size,mode="full")
    coarse_lag_idx = np.argmax(correlation)
    
    # Fit parabola.
    trim_lags = lags[coarse_lag_idx-tfit:coarse_lag_idx+tfit+1]
    trim_norm_cc = correlation[coarse_lag_idx-tfit:coarse_lag_idx+tfit+1]

    # Normalize.
    trim_norm_cc -= np.min(trim_norm_cc)
    trim_norm_cc /= np.max(trim_norm_cc)

    # Fit a polynomial and get the shift.
    p_coeffs = np.polyfit(trim_lags,trim_norm_cc,deg=2)
    lag_parab_hr = -p_coeffs[1]/(2*p_coeffs[0])*hrf
    
    return lag_parab_hr + tspc