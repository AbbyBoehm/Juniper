import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.cleaning import median_spatial_filter, colbycol_bckg, get_trace_mask, get_com_mask

def subtract_background(segments, inpt_dict):
    """Performs background subtraction on every integration in segments according to the instructions in inpt_dict.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        segments (dict): Its segments["data"] object will have its background removed.
        inpt_dict (dict): A dictionary containing instructions for performing this step.

    Returns:
        dict: segments["data"] with background removed.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Integration-level background subtraction processing...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Prep for plotting in case we do so.
    if (plot_step or save_step):
        # Create symlognorm color map.
        lin_threshold = 0.1
        vmin, vmax = np.nanpercentile(segments["data"][:,:,:],q=5), np.nanpercentile(segments["data"][:,:,:],q=95)
        symlog_norm_trace = colors.SymLogNorm(linthresh=lin_threshold, 
                                              linscale=1, 
                                              vmin=vmin,
                                              vmax=vmax,
                                              base=10)
        vmin, vmax = np.nanpercentile(segments["data"][:,:,:],q=0), np.nanpercentile(segments["data"][:,:,:],q=82)
        symlog_norm_bckgs = colors.SymLogNorm(linthresh=lin_threshold, 
                                              linscale=1, 
                                              vmin=vmin,
                                              vmax=vmax,
                                              base=10)

    # Obtain the mask that hides the trace using the median frame.
    trace_mask = np.zeros_like(segments["data"][0,:,:])
    if inpt_dict["trace_mask"]:
        median_integration = np.median(segments["data"],axis=0)
        if inpt_dict["trace_com_mask"]:
            trace_mask = get_com_mask(median_spatial_filter(median_integration,
                                                            sigma=inpt_dict["bckg_sigma"],
                                                            kernel=inpt_dict["bckg_kernel"]),
                                      width=inpt_dict["trace_com_mask"],
                                      upper=inpt_dict["trace_upw_mask"])
        else:
            trace_mask = get_trace_mask(median_spatial_filter(median_integration,
                                                              sigma=inpt_dict["bckg_sigma"],
                                                              kernel=inpt_dict["bckg_kernel"]),
                                        threshold=inpt_dict["bckg_threshold"])
        
        if (plot_step or save_step):
            # Create a diagnostic plot of the mask applied to the data.
            trace_mask_inverse = np.ma.masked_array(trace_mask,mask=np.ones_like(trace_mask)-trace_mask)
            fig, ax = plt.subplots(figsize=(20, 4))
            im = ax.imshow(median_integration,cmap='viridis',origin='lower',
                           norm=symlog_norm_bckgs,aspect='auto')
            ax.imshow(trace_mask_inverse,cmap='binary_r',origin='lower',
                      norm=symlog_norm_bckgs,aspect='auto')
            cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=40)
            cbar.set_label("Flux [DN]")
            ax.set_title("Integration-level 1/f trace mask")

            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs_trace_mask.png"),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()
        
    # Track backgrounds for plotting, and keep a frame handy for plotting.
    backgrounds = np.empty_like(segments["data"][:,:,:])
    precorrected_data = np.copy(segments["data"][:,:,:])

    # Background is already cleaned by this stage.
    bckg = np.copy(segments["data"][:,:,:])
            
    # Iterate over frames.
    for i in tqdm(range(segments["data"].shape[0]),
                  desc = "Removing 1/f noise from calibrated integrations...",
                  disable=(not time_ints)): # for each integration
        # Correct 1/f noise with integration-level background subtraction for that integration.
        segments["data"][i,:,:], backgrounds[i,:,:] = colbycol_bckg(segments["data"][i,:,:],
                                                                    bckg[i,:,:],
                                                                    inpt_dict["bckg_rows"],
                                                                    trace_mask)
        
    if (plot_step or save_step):
        # Create a diagnostic plot of the first integration's last group's residuals.
        median_integration = np.median(segments["data"],axis=0)

        fig, ax = plt.subplots(figsize=(20,12),nrows=3)
        fig.subplots_adjust(hspace=0.01,wspace=0.01)
        im1 = ax[0].imshow(precorrected_data[0,:,:],cmap='viridis',origin='lower',
                        norm=symlog_norm_trace,aspect='auto')
        im2 = ax[1].imshow(segments["data"][0,:,:],cmap='viridis',origin='lower',
                        norm=symlog_norm_trace,aspect='auto')
        im3 = ax[2].imshow(precorrected_data[0,:,:]-segments["data"][0,:,:],cmap='viridis',origin='lower',
                        norm='linear',aspect='auto')
        cbar = plt.colorbar(mappable=im3,orientation='horizontal',aspect=40)
        cbar.set_label("Stripe Flux [DN]")
        ax[0].set_title("Pre-corrected frame")
        ax[1].set_title("Corrected frame")
        ax[2].set_title("Stripes removed from frame")
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs_correction_example.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

        # Create a diagnostic plot of the median integration's background overlaid onto the data.
        median_background = np.median(np.copy(backgrounds),axis=0)
        background_mask = np.ones_like(median_background)
        if inpt_dict["trace_mask"]:
            background_mask[trace_mask==0] = 0
        else:
            minr, maxr = np.min(inpt_dict["bckg_rows"]), np.max(inpt_dict["bckg_rows"])
            if minr < 0:
                ubound, lbound = background_mask.shape[0] + minr, maxr
            else:
                lbound, ubound = minr, maxr
            background_mask[0:lbound,:] = 0
            background_mask[ubound:,:] = 0
        masked_background=np.ma.masked_array(median_background,mask=background_mask)

        fig, ax = plt.subplots(figsize=(20, 4))
        im1 = ax.imshow(median_integration,cmap='viridis',origin='lower',
                        norm=symlog_norm_trace,aspect='auto')
        im2 = ax.imshow(masked_background,cmap='viridis',origin='lower',
                        norm=symlog_norm_bckgs,aspect='auto')
        cbar1 = plt.colorbar(mappable=im1,orientation='horizontal',aspect=40)
        cbar1.set_label("Trace Flux [DN]")
        cbar2 = plt.colorbar(mappable=im2,orientation='horizontal',aspect=40)
        cbar2.set_label("Bckg Flux [DN]")
        ax.set_title("ILBS median background")
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs_median_background.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

        # Plot a 1D tseries of the median column bckg level in each frame.
        bckg_tseries = np.empty(backgrounds.shape[0])
        for i in range(backgrounds.shape[0]):
            # Extract bckg from raw data
            data = np.copy(precorrected_data[i,:,:])
            trace_mask[np.isnan(data)] = 1
            background_region = np.ma.masked_array(data,mask=trace_mask)
            if inpt_dict["bckg_rows"]:
                background_region = background_region[inpt_dict["bckg_rows"], :]
            bckg_tseries[i] = np.ma.median(background_region)
        
        fig, ax = plt.subplots(figsize=(12,3))
        ax.scatter(segments["time"],bckg_tseries,color='red',alpha=0.5)
        ax.plot(segments["time"],bckg_tseries,color='red',ls='--')

        ax.set_xlabel("Exposure Time [BJD TDB]")
        ax.set_ylabel("Flux [DN]")
        ax.set_title("Median Background Levels Before Correction")
        ax.tick_params(which='both',axis='both',direction='in')
        
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs_median_bckglevel.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

    if (plot_ints or save_ints):
        # Plot two 2D images of the columnal bckg level in each frame.
        bckg_tseries = np.empty((backgrounds.shape[0],backgrounds.shape[2])) # nints x ncols
        for i in range(backgrounds.shape[0]):
            # Extract bckg from raw data
            data = np.copy(precorrected_data[i,:,:])
            trace_mask[np.isnan(data)] = 1
            background_region = np.ma.masked_array(data,mask=trace_mask)
            if inpt_dict["bckg_rows"]:
                background_region = background_region[inpt_dict["bckg_rows"], :] # selected rows x all cols
            bckg_tseries[i,:] = np.ma.median(background_region,axis=0) # collapse on rows, yields shape ncols

        vmin, vmax = np.nanpercentile(bckg_tseries,q=5), np.nanpercentile(bckg_tseries,q=95)

        fig, ax = plt.subplots(figsize=(20,4))
        im1 = ax.imshow(bckg_tseries,origin='lower',cmap='viridis',
                        norm='linear',vmin=vmin,vmax=vmax,aspect='auto')
        cbar = plt.colorbar(mappable=im1,location='bottom',aspect=40)
        cbar.set_label('Flux [DN]')
        ax.set_xlabel('Column Index [#]')
        ax.set_ylabel('Integration Number [#]')
        ax.set_title("Median columnal background level")
        
        if save_ints:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_ilbs_median_bckgcolumnallevel.png"),
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