import os
from tqdm import tqdm
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
from astropy.io import fits

from juniper.stage1.wrap_stage1jwst import wrap_front_end
from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.plotting import img

def miri_refpix(datamodel, inpt_dict, plot_dir, outfile):
    """Performs MIRI reference pixel correction for an LRS array, since this is skipped by default \
    for subarrays by JWST pipeline.

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
        print("MIRI LRS reference pixel step processing...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Iterate over frames.
    for i in tqdm(range(datamodel.data.shape[0]),
                  desc = "Applying reference pixel correction of non-uniform readnoise from integrations...",
                  disable=(not time_step)): # for each integration
        # Grab copy of first group, we will need it.
        firstgroup = np.copy(datamodel.data[i,0,:,:])
        # Iterate over groups.
        odd_refs, even_refs, group_ns = [], [], []
        for g in tqdm(range(1,datamodel.data.shape[1]),
                      desc = "Correcing integration {}...".format(i),
                      disable=(not time_ints)): # for each group
            if (plot_step or save_step) and g == 1 and i == 0:
                # Plot and/or save the pre-sub first int's first group as an example.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect='auto', title="Group {}, int {} before refpix".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miri_refpix_pre-corr_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()
            if (plot_ints or save_ints):
                # Plot and/or save every pre-sub group to be thorough.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect='auto', title="Group {}, int {} before refpix".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_ints:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miri_refpix_pre-corr_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show()
                plt.close()

            # Subtract the first group out.
            datamodel.data[i,g,:,:] -= firstgroup

            # Calculate left means, with odd-even-rows as requested.
            if inpt_dict["odd_even_rows"]:
                # Fetch odd and even left data.
                left_odd = datamodel.data[i,g,1::2,:4]
                left_even = datamodel.data[i,g,0::2,:4]

                # Get the sigma-clipped means.
                left_odd_clipped, _, _ = sigmaclip(left_odd,3,3)
                odd_ref = np.mean(left_odd_clipped)

                left_even_clipped, _, _ = sigmaclip(left_even,3,3)
                even_ref = np.mean(left_even_clipped)

                # Subtract even_ref from even rows, and odd_ref from odd rows.
                datamodel.data[i,g,0::2,:] -= even_ref
                datamodel.data[i,g,1::2,:] -= odd_ref

                odd_refs.append(odd_ref)
                even_refs.append(even_ref)
            else:
                # Fetch left data.
                left_ref = datamodel.data[i,g,:,:4]

                # Get the sigma-clipped means.
                left_clipped, _, _ = sigmaclip(left_ref,3,3)
                ref = np.mean(left_clipped)

                # Subtract ref from all rows.
                datamodel.data[i,g,:,:] -= ref

                odd_refs.append(ref)
            group_ns.append(g)

            # Then add firstgroup back in.
            datamodel.data[i,g,:,:] += firstgroup
            
            if (plot_step or save_step) and g == 1 and i == 0:
                # Plot and/or save the post-sub first int's first group as an example.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect='auto', title="Group {}, int {} after refpix".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miri_refpix_post-corr_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()

            if (plot_ints or save_ints):
                # Plot and/or save every post-sub group to be thorough.
                fig, ax, im = img(datamodel.data[i,g,:,:], aspect='auto', title="Group {}, int {} after refpix".format(g, i),
                                  norm='linear',verbose=inpt_dict["verbose"])
                if save_ints:
                    plt.savefig(os.path.join(plot_dir,"S1_{}_miri_refpix_post-corr_g{}_i{}.png".format(outfile,g,i)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show()
                plt.close()

        if (((plot_step or save_step) and i == 0) or ((plot_ints or save_ints))):
            # Plot the references (odd/even if applicable)
            fig, ax = plt.subplots(figsize=(10,4))
            odd_label = 'odd/even references'
            if inpt_dict["odd_even_rows"]:
                odd_label = 'odd references'
            
            ax.plot(group_ns,odd_refs,color='k',ls='-',markerfacecolor='red',markeredgecolor='k',
                    marker='o',label=odd_label)
            if even_refs:
                ax.plot(group_ns,even_refs,color='k',ls='--',markerfacecolor='blue',markeredgecolor='k',
                        marker='o',label='even references')
            ax.set_xlabel("Group Number [#]")
            ax.set_ylabel("Reference Pixel Value [DN]")
            ax.legend()
            ax.tick_params(which='both',direction='in',axis='both')
            if save_step:
                plt.savefig(os.path.join(plot_dir,"S1_{}_miri_refpix_pix-values_i{}.png".format(outfile,i)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("MIRI LRS refpix correction complete.")
    return datamodel
