import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def clean_spec(oneD_spec, inpt_dict):
    """Compares all 1D spectra to a median spectra and replaces outliers
    with the median of that spectral point in time.

    Args:
        oneD_spec (np.array): 1D spectra to have outliers trimmed from.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array: 1D spectra with outliers cleaned.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Cleaning 1D spectrum for outliers...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Load sigma, and track outliers removed, and reserve pre-cleaned data.
    sigma = inpt_dict["sigma"]
    bad_spex_removed = 0
    pre_cleaned_spec = np.copy(oneD_spec)

    # Do not get stuck endlessly iterating - either stop at N_lim,
    # or stop when no more are found, whichever happens first.
    N_lim = 100
    iter_hit = N_lim
    outlier_found = True
    
    # Iterate cleaning.
    for i in tqdm(range(N_lim),
                  desc='Cleaning spectral outliers...',
                  disable=(not time_ints)):
        
        if outlier_found:
            # Define median spectrum in time, extend to shape of oneD_spec.
            med_spec = np.array([np.median(oneD_spec,axis=0),]*oneD_spec.shape[0])
            # Get standard deviation of each point, extend to shape of oneD_spec.
            std_spec = np.array([np.std(oneD_spec,axis=0),]*oneD_spec.shape[0])

            # Flag outliers.
            S = np.where(np.abs(oneD_spec-med_spec) > sigma*std_spec, 1, 0)

            # Count outliers found.
            bad_spex_this_step = np.count_nonzero(S)
            bad_spex_removed += bad_spex_this_step

            if bad_spex_this_step == 0:
                # No more outliers found! We can break the loop now.
                outlier_found = False
                iter_hit = i
            
            else:
                # Correct outliers and loop once more.
                oneD_spec = np.where(S == 1, med_spec, oneD_spec)
    if inpt_dict["verbose"] >= 1:
        print(f"Cleaning iteration stopped after {iter_hit+1} iterations.")

    if (plot_step or save_step):
        # Plot the median cleaned spectrum.
        fig,ax = plt.subplots(figsize = (10,5))
        ax.plot(np.median(pre_cleaned_spec,axis=0),color='midnightblue',alpha=0.75,ls='-',label='pre-correction')
        ax.plot(np.median(oneD_spec,axis=0),color='orange',alpha=0.75,ls='--',label='post-correction')
        ax.set_xlabel('Position [pix]')
        ax.set_ylabel('Flux [DN]')
        ax.legend(loc='upper right')
        ax.set_title('Pre- and post-cleaning median spectrum')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_cleaned_spec_median.png'),
                        dpi=300,bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    if (plot_ints or save_ints):
        # Plot all of the cleaned spectra on top of each other.
        fig,ax = plt.subplots(figsize = (10,5))
        ax.plot(pre_cleaned_spec[0,:],color='midnightblue',alpha=0.75,ls='-',label='pre-correction')
        ax.plot(oneD_spec[0,:],color='orange',alpha=0.75,ls='--',label='post-correction')
        for i in range(1,oneD_spec.shape[0]):
            ax.plot(pre_cleaned_spec[i,:],color='midnightblue',alpha=0.75,ls='-')
            ax.plot(oneD_spec[i,:],color='orange',alpha=0.75,ls='--')
        ax.set_xlabel('Position [pix]')
        ax.set_ylabel('Flux [DN]')
        ax.legend(loc='upper right')
        ax.set_title('Pre- and post-cleaning spectra')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_cleaned_spec_all.png'),
                        dpi=300,bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Spectra cleaned of outliers.")
    
    if inpt_dict["verbose"] == 2:
        print("Removed %.0f spectral outliers from spectra." % bad_spex_removed)
    
    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return oneD_spec

def clean_photseries(phot_tseries, phot_time, inpt_dict):
    """Takes a running median filter of the photometric time-series and clips outliers.

    Args:
        phot_tseries (np.array): photometric time-series to have outliers trimmed from.
        phot_time (np.array): timestamps for time-series, used for plotting.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array: photometric time-series with outliers cleaned.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Cleaning photometric time-series for outliers...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Load sigma, and track outliers removed, and reserve pre-cleaned data.
    sigma = inpt_dict["sigma"]
    bad_points_removed = 0
    pre_cleaned_tseries = np.copy(phot_tseries)

    # Do not get stuck endlessly iterating - either stop at N_lim,
    # or stop when no more are found, whichever happens first.
    N_lim = 100
    iter_hit = N_lim
    outlier_found = True
    
    # Iterate cleaning.
    for i in tqdm(range(N_lim),
                  desc='Cleaning time-series outliers...',
                  disable=(not time_ints)):
        
        if outlier_found:
            # Calculate running median time-series.
            runmed = medfilt(phot_tseries,kernel_size=inpt_dict["reject_window"]+1)
            
            # Get standard deviation of tseries.
            std = np.std(phot_tseries)

            # Flag outliers.
            S = np.where(np.abs(phot_tseries-runmed) > sigma*std, 1, 0)

            # Count outliers found.
            bad_points_this_step = np.count_nonzero(S)
            bad_points_removed += bad_points_this_step

            if bad_points_this_step == 0:
                # No more outliers found! We can break the loop now.
                outlier_found = False
                iter_hit = i
            
            else:
                # Correct outliers and loop once more.
                phot_tseries = np.where(S == 1, runmed, phot_tseries)
    if inpt_dict["verbose"] >= 1:
        print(f"Cleaning iteration stopped after {iter_hit+1} iterations.")

    if (plot_step or save_step):
        # Plot the cleaned photometric time-series.
        fig,ax = plt.subplots(figsize = (10,5))
        ax.plot(phot_time,pre_cleaned_tseries,color='midnightblue',marker='s',alpha=0.75,ls='-',label='pre-correction')
        ax.plot(phot_time,phot_tseries,color='orange',marker='o',alpha=0.75,ls='-',label='post-correction')
        ax.set_xlabel('Exposure Time [BJD TDB]')
        ax.set_ylabel('Flux [DN]')
        ax.legend(loc='upper right')
        ax.set_title('Pre- and post-cleaning photometric time-series')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_cleaned_time-series.png'),
                        dpi=300,bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Photometric time-series cleaned of outliers.")
    
    if inpt_dict["verbose"] == 2:
        print("Removed %.0f outliers from time-series." % bad_points_removed)
    
    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return phot_tseries