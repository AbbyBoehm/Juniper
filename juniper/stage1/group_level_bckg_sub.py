import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.cleaning import median_spatial_filter, colbycol_bckg, get_trace_mask, get_com_mask
from juniper.util.plotting import img

def glbs(datamodel, inpt_dict, plot_dir, outfile):
    """Performs group-level background subtraction on every group in the datamodel according to the instructions in inpt_dict.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        datamodel (jwst.datamodel): A datamodel containing attribute .data, which is an np array of shape nints x ngroups x nrows x ncols, produced during wrap_front_end.
        inpt_dict (dict): A dictionary containing instructions for performing this step.
        plot_dir (str): location to save diagnostic plots to.
        outfile (str): helps keep diagnostic plots distinct.

    Returns:
        jwst.datamodel: datamodel with updated cleaned .data attribute.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Group-level background subtraction processing...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Copy data.
    data = np.copy(datamodel.data)

    # Prep for plotting in case we do so.
    if (plot_step or save_step):
        # Create symlognorm color map.
        lin_threshold = 0.1
        vmin, vmax = np.percentile(data[:,-1,:,:],q=5), np.percentile(data[:,-1,:,:],q=95)
        symlog_norm_trace = colors.SymLogNorm(linthresh=lin_threshold, 
                                            linscale=1, 
                                            vmin=vmin,
                                            vmax=vmax,
                                            base=10)
        vmin, vmax = np.percentile(data[:,-1,:,:],q=0), np.percentile(data[:,-1,:,:],q=82)
        symlog_norm_bckgs = colors.SymLogNorm(linthresh=lin_threshold, 
                                            linscale=1, 
                                            vmin=vmin,
                                            vmax=vmax,
                                            base=10)

    # If needed, get the trace mask using the median last group.
    trace_mask = np.zeros_like(data[0,0,:,:])
    if inpt_dict["mask"]:
        median_last_group = np.median(np.copy(data[:,-1,:,:]),axis=0)
        if inpt_dict["com_mask"]:
            trace_mask = get_com_mask(median_spatial_filter(median_last_group,
                                                            sigma=inpt_dict["sigma"],
                                                            kernel=inpt_dict["kernel"]),
                                                            width=inpt_dict["com_mask"],
                                                            contrast=inpt_dict["low_contr"])
        else:
            trace_mask = get_trace_mask(median_spatial_filter(median_last_group,
                                                              sigma=inpt_dict["sigma"],
                                                              kernel=inpt_dict["kernel"]),
                                                              threshold=inpt_dict["threshold"])
        if (plot_step or save_step):
            # Create a diagnostic plot of the mask applied to the data.
            trace_mask_inverse = np.ma.masked_array(trace_mask,mask=np.ones_like(trace_mask)-trace_mask)
            fig, ax = plt.subplots(figsize=(20, 4))
            im = ax.imshow(median_last_group,cmap='viridis',origin='lower',
                           norm=symlog_norm_bckgs,aspect='auto')
            ax.imshow(trace_mask_inverse,cmap='binary_r',origin='lower',
                      norm=symlog_norm_bckgs,aspect='auto')
            cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=40)
            cbar.set_label("Flux [DN]")
            ax.set_title("Trace mask applied to data")

            if save_step:
                plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_trace_mask.png".format(outfile)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()

    # Track backgrounds for plotting, and keep a frame handy for plotting.
    backgrounds = np.empty_like(datamodel.data[:,:,:,:])
    precorrected_frame = np.copy(datamodel.data[0,-1,:,:])
    # Iterate over frames.
    for i in tqdm(range(data.shape[0]),
                  desc = "Removing 1/f noise from integrations...",
                  disable=(not time_step)): # for each integration    
        for g in range(data.shape[1]): # for each group
            # Correct 1/f noise with group-level background subtraction for that group.
            datamodel.data[i,g,:,:], backgrounds[i,g,:,:] = colbycol_bckg(data[i,g,:,:],
                                                                          inpt_dict["rows"],
                                                                          trace_mask)
        
    ngroup = data.shape[1]
    if (plot_step or save_step):
        # Create a diagnostic plot of the first integration's last group's residuals.
        fig, ax = plt.subplots(figsize=(20,12),nrows=3)
        im1 = ax[0].imshow(precorrected_frame,cmap='viridis',origin='lower',
                           norm=symlog_norm_trace,aspect='auto')
        im2 = ax[1].imshow(datamodel.data[0,-1,:,:],cmap='viridis',origin='lower',
                           norm=symlog_norm_trace,aspect='auto')
        im3 = ax[2].imshow(precorrected_frame-datamodel.data[0,-1,:,:],cmap='viridis',origin='lower',
                           norm='linear',aspect='auto')
        cbar = plt.colorbar(mappable=im3,orientation='horizontal',aspect=40)
        cbar.set_label("Stripe Flux [DN]")
        ax[0].set_title("Pre-corrected frame")
        ax[1].set_title("Corrected frame")
        ax[2].set_title("Stripes removed from frame")
        if save_step:
            plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_correction_example.png".format(outfile)),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

        # Create a diagnostic plot of the median last group's background overlaid onto the data.
        median_background = np.median(np.copy(backgrounds[:,-1,:,:]),axis=0)
        background_mask = np.ones_like(median_background)
        if inpt_dict["mask"]:
            background_mask[trace_mask==0] = 0
        else:
            minr, maxr = np.min(inpt_dict["rows"]), np.max(inpt_dict["rows"])
            if minr < 0:
                ubound, lbound = background_mask.shape[0] + minr, maxr
            else:
                lbound, ubound = minr, maxr
            background_mask[0:lbound,:] = 0
            background_mask[ubound:,:] = 0
        masked_background=np.ma.masked_array(median_background,mask=background_mask)

        fig, ax = plt.subplots(figsize=(20, 4))
        im1 = ax.imshow(median_last_group,cmap='viridis',origin='lower',
                        norm=symlog_norm_trace,aspect='auto')
        im2 = ax.imshow(masked_background,cmap='viridis',origin='lower',
                        norm=symlog_norm_bckgs,aspect='auto')
        cbar1 = plt.colorbar(mappable=im1,orientation='horizontal',aspect=40)
        cbar1.set_label("Trace Flux [DN]")
        cbar2 = plt.colorbar(mappable=im2,orientation='horizontal',aspect=40)
        cbar2.set_label("Bckg Flux [DN]")
        ax.set_title("GLBS median background")
        if save_step:
            plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_median_background_group{}.png".format(outfile,ngroup)),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

    if (plot_ints or save_ints):
        # Create a plot of the median background at each group level.
        for g in tqdm(range(ngroup),
                      desc='Creating GLBS group plots...'):
            median_group = np.median(np.copy(data[:,g,:,:]),axis=0)
            median_background = np.median(np.copy(backgrounds[:,g,:,:]),axis=0)
            background_mask = np.ones_like(median_background)
            if inpt_dict["mask"]:
                background_mask[trace_mask==0] = 0
            else:
                minr, maxr = np.min(inpt_dict["rows"]), np.max(inpt_dict["rows"])
                if minr < 0:
                    ubound, lbound = background_mask.shape[0] + minr, maxr
                else:
                    lbound, ubound = minr, maxr
                background_mask[0:lbound,:] = 0
                background_mask[ubound:,:] = 0
            masked_background=np.ma.masked_array(median_background,mask=background_mask)

            fig, ax = plt.subplots(figsize=(20, 4))
            im1 = ax.imshow(median_group,cmap='viridis',origin='lower',
                            norm=symlog_norm_trace,aspect='auto')
            im2 = ax.imshow(masked_background,cmap='viridis',origin='lower',
                            norm=symlog_norm_bckgs,aspect='auto')
            cbar1 = plt.colorbar(mappable=im1,orientation='horizontal',aspect=40)
            cbar1.set_label("Trace Flux [DN]")
            cbar2 = plt.colorbar(mappable=im2,orientation='horizontal',aspect=40)
            cbar2.set_label("Bckg Flux [DN]")
            ax.set_title("GLBS median background")

            if save_step:
                plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_median_background_group{}.png".format(outfile,g+1)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()
    
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Group-level background subtraction complete.")
    return datamodel