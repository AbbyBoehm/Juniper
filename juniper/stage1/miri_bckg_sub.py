import os
from tqdm import tqdm
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from juniper.stage1.wrap_stage1jwst import wrap_front_end
from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.plotting import img

def miribckg(datamodel, steps, inpt_dict, plot_dir, outfile):
    """Performs group-level background subtraction on every group in the datamodel according to the instructions in inpt_dict.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        datamodel (jwst.datamodel): A datamodel containing attribute .data, which is an np array of shape nints x ngroups x nrows x ncols, produced during wrap_front_end.
        steps (dict): The full dictionary of instructions, needed to reprocess the bckg.
        inpt_dict (dict): A dictionary containing instructions for performing this step.
        plot_dir (str): location to save diagnostic plots to.
        outfile (str): helps keep diagnostic plots distinct.

    Returns:
        jwst.datamodel: datamodel with updated cleaned .data attribute.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("MIRI LRS group-by-group background subtraction processing...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Load in the MIRI LRS background image.
    try:
        bckg = wrap_front_end(inpt_dict['path'],steps)
        bckg = bckg.data
        bckg = np.median(bckg,axis=0) # axis 0 is integrations
        if (plot_step or save_step):
                # Save a diagnostic plot of the first trace mask.
                fig, ax, im = img(bckg[0,:,:], aspect=5, title="Int 0 bckg",
                                  norm='log',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miribckg_example.png".format(outfile)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()
    except FileNotFoundError:
        warn("FileNotFoundError encountered for MIRI LRS bckg path: ",inpt_dict['miri_bckg'])
        print("MIRI LRS bckg correction will be skipped. Please update the filepath if you wish to perform bckg subtraction.")
        return

    # Iterate over frames.
    for i in tqdm(range(datamodel.data.shape[0]),
                  desc = "Removing 1/f noise from integrations...",
                  disable=(not time_step)): # for each integration
        # Iterate over groups.
        for g in tqdm(range(datamodel.data.shape[1]),
                      desc = "Correcing integration {}...".format(i),
                      disable=(not time_ints)): # for each group
            if (plot_step or save_step) and g == 0 and i == 0:
                # Plot and/or save the pre-sub first int's first group as an example.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect=5, title="Group {}, int {} before sub".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miribckg_pre-sub_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()
            if (plot_ints or save_ints):
                # Plot and/or save every pre-sub group to be thorough.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect=5, title="Group {}, int {} before sub".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_ints:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miribckg_pre-sub_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show()
                plt.close()

            # Subtract the bckg from the data.
            datamodel.data[i,g,:,:] -= bckg[g,:,:]
            
            if (plot_step or save_step) and g == 0 and i == 0:
                # Plot and/or save the post-sub first int's first group as an example.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect=5, title="Group {}, int {} after sub".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miribckg_post-sub_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()
            if (plot_ints or save_ints):
                # Plot and/or save every post-sub group to be thorough.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect=5, title="Group {}, int {} after sub".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_ints:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miribckg_post-sub_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show()
                plt.close()
    
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("MIRI LRS group-by-group background subtraction complete.")
    return datamodel
