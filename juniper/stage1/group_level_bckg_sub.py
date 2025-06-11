import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

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
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Copy data.
    data = np.copy(datamodel.data)

    # Iterate over frames.
    for i in tqdm(range(data.shape[0]),
                  desc = "Removing 1/f noise from integrations...",
                  disable=(not time_step)): # for each integration
        # Obtain the mask that hides the trace using a cleaned version of the very last group in this integration.
        trace_mask = np.zeros_like(data[i,-1,:,:])
        if inpt_dict["mask"]:
            if inpt_dict["com_mask"]:
                trace_mask = get_com_mask(median_spatial_filter(np.copy(data[i,-1,:,:]),
                                                                sigma=inpt_dict["sigma"],
                                                                kernel=inpt_dict["kernel"]),
                                          width=inpt_dict["com_mask"])
            else:
                trace_mask = get_trace_mask(median_spatial_filter(np.copy(data[i,-1,:,:]),
                                                                  sigma=inpt_dict["sigma"],
                                                                  kernel=inpt_dict["kernel"]),
                                            threshold=inpt_dict["threshold"])
            
            if (plot_step or save_step) and i == 0:
                # Save a diagnostic plot of the first trace mask.
                fig, ax, im = img(trace_mask, aspect=5, title="1/f trace mask",
                                  vmin=0, vmax=1, norm='linear',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_trace_mask_{}.png".format(outfile, i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()
            if (plot_ints or save_ints):
                # Save a diagnostic plot of every trace mask.
                fig, ax, im = img(trace_mask, aspect=5, title="1/f trace mask",
                                  vmin=0, vmax=1, norm='linear',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_trace_mask_{}.png".format(outfile, i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()
            
        for g in tqdm(range(data.shape[1]),
                      desc = "Correcing integration {}...".format(i),
                      disable=(not time_ints)): # for each group
            # Correct 1/f noise with group-level background subtraction for that group.
            datamodel.data[i,g,:,:], background = colbycol_bckg(data[i,g,:,:],
                                                                inpt_dict["rows"],
                                                                trace_mask)
            
            if (plot_step or save_step) and g == 0 and i == 0:
                # Plot and/or save the background of the first int's first group as an example.
                fig, ax, im = img(background, aspect=5, title="Group {}, int {} background".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_bckg_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()
            if (plot_ints or save_ints):
                # Plot and/or save every background to be thorough.
                fig, ax, im = img(background, aspect=5, title="Group {}, int {} background".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_ints:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_glbs_bckg_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show()
                plt.close()
    
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Group-level background subtraction complete.")
    return datamodel