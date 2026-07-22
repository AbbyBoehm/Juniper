import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d

from jwst.datamodels import dqflags

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import aspect_handler

def mask_flags(segments, inpt_dict):
    """Uses the jwst pipeline data quality flags to mask bad pixels.

    Args:
        segments (dict): Its segments["data"] object contains the integrations \
        and segments["jwstdq"] contains the data quality flags.
        inpt_dict (dict): instructions for how to process this step.

    Returns:
        dict: segments with the flagged pixels masked or corrected.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Treating pixels flagged by the jwst pipeline...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time this step, if asked.
    if time_step:
        t0 = time.time()

    # Have all JWST flags on hand for reference
    all_jwst_flags = [2**i for i in range(1,32)]

    # Create plotting copy of first frame.
    if (plot_step or save_step):
        precorrected_frame = segments["data"][0,:,:]

    # Protect certain flags, if asked.
    if (inpt_dict["skip_flags"] and not inpt_dict["target_flags"]):
        for flag in tqdm(inpt_dict["skip_flags"],
                         desc='Removing protected flags...',
                         disable=(not time_ints)):
            # Interpret the flag as a power of 2.
            thebit=dqflags.interpret_bit_flags(flag,mnemonic_map=dqflags.pixel)
            
            # Locate where the bit is.
            bitishere = locate_flag(thebit,segments["jwstdq"],all_jwst_flags)

            # As this flag is not to be considered, subtract it from the dq array.
            segments["jwstdq"][bitishere] -= thebit
    
    # Alternatively, only target certain flags.
    elif inpt_dict["target_flags"]:
        allthebit = np.zeros_like(segments["jwstdq"])
        for flag in tqdm(inpt_dict["target_flags"],
                         desc='Targeting selected flags...',
                         disable=(not time_ints)):
            # Interpret the flag as a power of 2.
            thebit=dqflags.interpret_bit_flags(flag,mnemonic_map=dqflags.pixel)

            # Locate where the bit is.
            bitishere = locate_flag(thebit,segments["jwstdq"],all_jwst_flags)
            allthebit[bitishere] = 1
        # If only these flags are to be targeted, 0 out everything else.
        segments["jwstdq"][allthebit!=1] = 0

    # Turn dqflags into mask arrays, and add nan mask.
    dq_mask = np.empty_like(segments["jwstdq"])
    dq_mask[:, :, :] = np.where(segments["jwstdq"] > 0, 1, 0)
    dq_mask[np.isnan(segments["data"])] = 1

    if (plot_step or save_step):
        # Create plots of the entire dq_mask collapsed in on itself in time.
        dq_alltime = np.sum(dq_mask,axis=0)
        fig, ax = plt.subplots(figsize=(20,4))
        img_aspect, cbar_aspect = aspect_handler(dq_alltime)
        im = ax.imshow(dq_alltime/dq_mask.shape[0],origin='lower',cmap='viridis',
                       norm='linear',aspect=img_aspect,vmin=0,vmax=1)
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Fraction of integrations flagged')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_JWST_flags-all.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create plots for each flag targeted.
        plot_these_flags = list(dqflags.pixel.keys())
        if (inpt_dict["skip_flags"] and not inpt_dict["target_flags"]):
            plot_these_flags = [x for x in plot_these_flags if x not in inpt_dict["skip_flags"]]
        elif inpt_dict["target_flags"]:
            plot_these_flags = [x for x in plot_these_flags if x in inpt_dict["target_flags"]]
        
        # Go through each flag and find where flag was assigned.
        for flag in tqdm(plot_these_flags,
                         desc='Plotting selected flags...',
                         disable=(not time_ints)):
            # Create template dq flag map.
            thisflag = np.zeros_like(segments["jwstdq"])

            # Interpret the flag as a power of 2.
            thebit=dqflags.interpret_bit_flags(flag,mnemonic_map=dqflags.pixel)

            # Locate where the bit is.
            bitishere = locate_flag(thebit,segments["jwstdq"],all_jwst_flags)

            # Set thisflag to 1 where the bit was found.
            thisflag[bitishere] = 1

            # Collapse in time.
            thisflagovertime = np.sum(thisflag,axis=0)

            # And plot!
            fig, ax = plt.subplots(figsize=(20,4))
            img_aspect, cbar_aspect = aspect_handler(thisflagovertime)
            im = ax.imshow(thisflagovertime/thisflag.shape[0],origin='lower',cmap='viridis',
                           norm='linear',aspect=img_aspect,vmin=0,vmax=1)
            cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
            cbar.set_label('Fraction of integrations flagged {}'.format(flag))

            if save_ints:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_JWST_flagged-for-{}.png".format(flag)),
                            dpi=300, bbox_inches='tight')
            if plot_ints:
                plt.show()
            plt.close()

    # The replacement method can be median time, median space, or None to leave as masked.
    if inpt_dict["flag_replace"] == 'time':
        if inpt_dict["verbose"] == 2:
            print("Replacing flagged pixels with the median in time...")
        # One filtered image as the median in time.
        replacement = np.median(segments["data"],axis=0)
        # And replace.
        segments["data"] = np.where(dq_mask > 0, replacement, segments["data"])

    elif inpt_dict["flag_replace"] == 'space':
        if inpt_dict["verbose"] == 2:
            print("Replacing flagged pixels with spatially median-filtered values...")
        # Produce a median-filtered spatial image for each frame.
        replacement = np.empty_like(segments["data"])
        for k in range(replacement.shape[0]):
            frame = segments["data"][k]
            frame[np.isnan(frame)] = 0
            frame = medfilt2d(frame,inpt_dict["flag_kernel"])
            replacement[k,:,:] = frame
        # And replace.
        segments["data"] = np.where(dq_mask > 0, replacement, segments["data"])

    # Otherwise, just mask the bad pixels.
    else:
        if inpt_dict["verbose"] == 2:
            print("Masking flagged pixels with the badpixmask attribute...")
        segments["badpixmask"] = np.where(dq_mask>0,1,segments["badpixmask"])

    # Show change in first frame after this action.
    if (plot_step or save_step):
        fig, ax = plt.subplots(2,1,figsize=(20,5),sharex=True)
        img_aspect, _ = aspect_handler(precorrected_frame)
        vmin, vmax = np.nanpercentile(precorrected_frame[np.isfinite(precorrected_frame)],q=5), np.nanpercentile(precorrected_frame[np.isfinite(precorrected_frame)],q=95)
        if vmin <= 0:
            pos = np.copy(precorrected_frame[precorrected_frame>0])
            vmin, vmax = np.nanpercentile(pos[np.isfinite(pos)],q=5), np.nanpercentile(pos[np.isfinite(pos)],q=95)
        ax[0].imshow(precorrected_frame,aspect=img_aspect,cmap='viridis',origin='lower',
                     vmin=vmin,vmax=vmax,norm='log')
        ax[0].set_title("Pre-correction integration 0")
        ax[1].imshow(segments["data"][0,:,:],aspect=img_aspect,cmap='viridis',origin='lower',
                     vmin=vmin,vmax=vmax,norm='log')
        ax[1].set_title("Post-correction integration 0")
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_JWST_flags-corrected_int0.png"),
                            dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    # Count how many pixels were replaced.
    if inpt_dict["verbose"] >= 1:
        print("{} flagged pixels were masked or replaced.".format(np.count_nonzero(dq_mask)))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    return segments

def locate_flag(flag,dq,all_jwst_flags):
    """Simple function to locate the flag in the dq array.

    Args:
        flag (int): the bit interpretation of the target flag.
        dq (array-like): the data quality array to search for the flag in.
        all_jwst_flags (lst): all flag bit values, for reference.
    """
    # Copy the array.
    search_dq = np.copy(dq)

    # Iterate over time.
    for i in (range(dq.shape[0])):
        # Fetch relevant data.
        dq_frame = search_dq[i]

        # Subtract off flags larger than the current flag if present.
        for jwst_flag in [x for x in sorted(all_jwst_flags,reverse=True) if x > flag]:
            dq_frame[dq_frame>=jwst_flag] -= jwst_flag
        
        # Anything greater than or equal to the desired flag at this point has our flag in it.
        dq_frame[dq_frame<flag] = 0

        search_dq[i,:,:] = dq_frame
    indx = np.where(search_dq != 0)
    return indx
