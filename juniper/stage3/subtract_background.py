import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.cleaning import median_spatial_filter, colbycol_bckg, get_trace_mask, get_com_mask
from juniper.util.plotting import img

def subtract_background(segments, inpt_dict):
    """Performs background subtraction on every integration in segments according to the instructions in inpt_dict.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        segments (xarray): Its segments.data object will have its background removed.
        inpt_dict (dict): A dictionary containing instructions for performing this step.

    Returns:
        xarray: segments.data with background removed.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Integration-level background subtraction processing...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Obtain the mask that hides the trace using the median frame.
    trace_mask = np.zeros_like(segments.data[0,:,:])
    if inpt_dict["trace_mask"]:
        if inpt_dict["trace_com_mask"]:
            trace_mask = get_com_mask(median_spatial_filter(np.median(segments.data.values,axis=0),
                                                            sigma=inpt_dict["bckg_sigma"],
                                                            kernel=inpt_dict["bckg_kernel"]),
                                      width=inpt_dict["trace_com_mask"])
        else:
            trace_mask = get_trace_mask(median_spatial_filter(np.median(segments.data.values,axis=0),
                                                            sigma=inpt_dict["bckg_sigma"],
                                                            kernel=inpt_dict["bckg_kernel"]),
                                        threshold=inpt_dict["bckg_threshold"])
        
        if (plot_step or save_step):
            # Save a diagnostic plot of the trace mask.
            fig, ax, im = img(trace_mask, aspect=5, title="Integration-level 1/f trace mask",
                                vmin=0, vmax=1, norm='linear',verbose=inpt_dict["verbose"])
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs-trace-mask.png"),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()

    # Iterate over frames.
    for i in tqdm(range(segments.data.shape[0]),
                  desc = "Removing 1/f noise from calibrated integrations...",
                  disable=(not time_ints)): # for each integration
        # Correct 1/f noise with integration-level background subtraction for that integration.
        segments.data.values[i,:,:], background = colbycol_bckg(segments.data.values[i,:,:],
                                                                inpt_dict["bckg_rows"],
                                                                trace_mask)
        
        if (plot_step or save_step) and i == 0:
            # Plot and/or save the background of the first int as an example.
            fig, ax, im = img(background, aspect=5, title="Int {} background".format(i),
                                norm='log',verbose=inpt_dict["verbose"])
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs-bckg_int{}.png".format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()
        if (plot_ints or save_ints):
            # Plot and/or save every background to be thorough.
            fig, ax, im = img(background, aspect=5, title="Int {} background".format(i),
                                norm='log',verbose=inpt_dict["verbose"])
            if save_ints:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs-bckg_int{}.png".format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_ints:
                plt.show()
            plt.close()
    
    if inpt_dict["verbose"] >= 1:
        print("Integration-level background subtraction complete.")

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return segments