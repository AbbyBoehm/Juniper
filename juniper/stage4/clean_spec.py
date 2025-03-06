import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

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
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track cleaned spectra and load sigma.
    cleaned_specs = []
    sigma = inpt_dict["sigma"]

    # Track outliers removed.
    bad_spex_removed = 0
    
    # Iterate over spectra.
    for i in tqdm(range(oneD_spec.shape[0]),
                  desc='Cleaning spectral outliers...',
                  disable=(not time_ints)):
        # Iteration stop condition. As long as outliers are being found, we have to keep iterating.
        outlier_found = True
        # But also, no need to be stuck endlessly iterating.
        N_lim = 100
        n = 0
        while outlier_found:
            # Define median spectrum in time.
            med_spec = np.median(oneD_spec,axis=0)
            # Get standard deviation of each point.
            std_spec = np.std(oneD_spec,axis=0)

            # Flag outliers.
            S = np.where(np.abs(oneD_spec[i,:]-med_spec) > sigma*std_spec, 1, 0)

            # Count outliers found.
            bad_spex_this_step = np.count_nonzero(S)
            bad_spex_removed += bad_spex_this_step

            if bad_spex_this_step == 0:
                # No more outliers found! We can break the loop now.
                outlier_found = False
            
            # Correct outliers and loop once more.
            oneD_spec[i,:] = np.where(S == 1, med_spec, oneD_spec[i,:])

            n += 1
            if n > N_lim:
                # We are breaking the loop to save computing time.
                if inpt_dict["verbose"] >= 1:
                    print("Hit iteration limit on spec {}.".format(i))
                outlier_found = False
        cleaned_specs.append(oneD_spec[i,:])

        if (plot_step or save_step) and i==0:
            plt.plot(oneD_spec[i,:],color='midnightblue',alpha=0.5,ls='-')
            plt.plot(cleaned_specs[i],color='orange',alpha=0.5,ls='--')
            plt.xlabel('position [pix]')
            plt.ylabel('flux [a.u.]')
            plt.title('Cleaned spectrum 0')
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_cleaned_spec_0.png'),
                            dpi=300,bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

        if (plot_ints or save_ints):
            plt.plot(oneD_spec[i,:],color='midnightblue',alpha=0.5,ls='-')
            plt.plot(cleaned_specs[i],color='orange',alpha=0.5,ls='--')
            plt.xlabel('position [pix]')
            plt.ylabel('flux [a.u.]')
            plt.title('Cleaned spectrum {}'.format(i))
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_cleaned_spec_{}.png'.format(i)),
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

    return np.array(cleaned_specs)