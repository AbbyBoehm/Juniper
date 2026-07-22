import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import aspect_handler

def iterate_fixed(segments, inpt_dict):
    """Iterate a fixed number of times at specified sigmas to remove cosmic rays.

    Args:
        segments (dict): Its segments["data"] object has the integrations \
        and its junidq object contains the data quality flags to be updated.
        inpt_dict (dict): instructions for how to run this step.

    Returns:
        dict: segments with removed cosmic rays and data quality flags updated.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Iterating fixed number of times to remove cosmic rays...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found.
    bad_pix_map = np.zeros_like(segments["data"])

    # Track sigma differences for plotting.
    sig_diffs = np.empty_like(segments["data"])
    
    # Start iterating.
    for sigma in inpt_dict["fixed_sigmas"]:
        # Compute median image and std deviation.
        med = np.ma.median(segments["data"], axis=0)
        std = np.ma.std(segments["data"], axis=0)

        # Track bad pixels in this sigma.
        bad_pix_this_sigma = 0

        for k in tqdm(range(segments["data"].shape[0]),
                      desc='Iterating over sigma=%.2f...'%sigma,
                      disable=(not time_ints)):
            if sigma == inpt_dict["fixed_sigmas"][0]:
                # Retain sigma excess.
                sig_diffs[k,:,:] = np.abs(segments["data"][k,:,:] - med)/std

            # Look for where outliers are in this frame and flag with 1.
            S = np.where(np.abs(segments["data"][k,:,:] - med) > sigma*std, 1, 0)

            # Count outliers and locate them.
            S = np.where(bad_pix_map[k,:,:] == 1, 0, S) # if the pixel was already reported as bad from a previous step, then don't double count.
            bad_pix_this_sigma += np.count_nonzero(S)
            bad_pix_map[k,:,:] += S

            # If time_replace is not None, we replace the bad pixels instead of just masking them with the DQ array.
            if inpt_dict["time_replace"]:
                correction = med
                if inpt_dict["time_replace"] != 'all':
                    # Take the median of the frames that are +/- time_replace away from the current frame.
                    l = k - inpt_dict["time_replace"]
                    r = k + inpt_dict["time_replace"]
                    # Cut at edges.
                    if l < 0:
                        l = 0
                    if r > segments["data"].shape[0]:
                        r = segments["data"].shape[0]
                    correction = np.median(segments["data"][l:r,:,:],axis=0)
                # And replace with correction.
                segments["data"][k,:,:] = np.where(S == 1, correction, segments["data"][k,:,:])
            # Otherwise, just mask the bad pixels.
            else:
                segments["badpixmask"][k,:,:] = np.where(bad_pix_map[k,:,:]>0,1,
                                                         segments["badpixmask"][k,:,:])
            
        # Report how many bad pixels this sigma trimmed.
        if inpt_dict["verbose"] == 2:
            print("Bad pixels flagged at sigma=%.2f: %.0f"%(sigma, bad_pix_this_sigma))
    
    # Update data flags.
    segments["junidq"] = np.where(bad_pix_map != 0, 1, segments["junidq"])

    if (plot_step or save_step):
        # Create a plot of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        fig, ax = plt.subplots(figsize=(20,4))
        img_aspect, cbar_aspect = aspect_handler(bad_pix_alltime)
        im = ax.imshow(bad_pix_alltime/bad_pix_map.shape[0],origin='lower',cmap='viridis',
                       norm='linear',aspect=img_aspect,vmin=0,vmax=0.01)
        ax.set_title("Fixed-iteration DQ flags")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Fraction of integrations flagged')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-fixed_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create a plot of the maximum sigma difference that pixel felt.
        fig, ax = plt.subplots(figsize=(20,4))
        sig_diff = np.empty_like(sig_diffs[0,:,:])
        img_aspect, cbar_aspect = aspect_handler(sig_diff)
        for x1 in range(sig_diffs.shape[1]):
            for x2 in range(sig_diffs.shape[2]):
                sig_diff[x1,x2] = np.nanmax(sig_diffs[:,x1,x2])
        
        vmin, vmax = 0, 1.5*max(inpt_dict["fixed_sigmas"])
        im = ax.imshow(sig_diff,origin='lower',cmap='viridis',
                       norm='linear',aspect=img_aspect,vmin=vmin,vmax=vmax)
        ax.set_title("Maximum sigma difference from median")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Flux [DN]')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-fixed_biggest-differences.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Iterations complete. Total bad pixels found: %.0f."% np.count_nonzero(bad_pix_map))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return segments

def iterate_free(segments, inpt_dict):
    """Iterate an unspecified number of times at a fixed sigma to remove cosmic rays.

    Args:
        segments (dict): Its segments["data"] object has the integrations \
        and its junidq object contains the data quality flags to be updated.
        inpt_dict (dict): instructions for how to run this step.

    Returns:
        dict: segments with removed cosmic rays and data quality flags updated.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Iterating free number of times to remove cosmic rays...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found, and open sigma once.
    bad_pix_map = np.zeros_like(segments["data"])
    sigma = inpt_dict["free_sigma"]

    # Track sigma differences for plotting.
    sig_diffs = np.zeros_like(segments["data"])

    # Check force stop iteration condition.
    cut_off = np.inf
    if inpt_dict["free_cutoffs"]:
        cut_off = inpt_dict["free_cutoffs"]
    
    # Start iterating.
    for i in tqdm(range(segments["data"].shape[1]),
                  desc='Iterating over pixel time series...',
                  disable=(not time_ints)):
        for j in range(segments["data"].shape[2]):
            # Claim outlier found and track steps.
            outliers_found = 1
            step = 0
            
            # Then, iterate until no outliers found, or until cut off kicks in.
            while (outliers_found > 0 and step < cut_off):
                # Compute median pixel time series and std deviation.
                med = np.median(segments["data"][:,i,j])
                std = np.std(segments["data"][:,i,j])

                # Look for a bigger sigma outlier.
                sig_diff = np.abs(segments["data"][:,i,j]-med)/std
                sig_diffs[sig_diff>sig_diffs[:,i,j],i,j] = sig_diff

                # Check for outliers.
                S = np.where(np.abs(segments["data"][:,i,j]-med) > sigma*std, 1, 0)

                # Count outliers and locate them.
                S = np.where(bad_pix_map[:,i,j] == 1, 0, S) # if the pixel was already reported as bad from a previous step, then don't double count.
                outliers_found = np.count_nonzero(S)
                bad_pix_map[:,i,j] += S

                # If time_replace is not None, we replace the bad pixels instead of just masking them with the DQ array.
                if inpt_dict["time_replace"]:
                    correction = med
                    if inpt_dict["time_replace"] != 'all':
                        # Smooth the pixel's time series over with a running median rather than a median in all time.
                        correction = medfilt(segments["data"][:,i,j], kernel_size=(2*inpt_dict["time_replace"]+1))
                    # And replace with correction.
                    segments["data"][:,i,j] = np.where(S == 1, correction, segments["data"][:,i,j])
                # Otherwise, just mask the bad pixels.
                else:
                    segments["badpixmask"][:,i,j] = np.where(bad_pix_map[:,i,j]>0,1,
                                                            segments["badpixmask"][:,i,j])

                # Advance another step.
                step += 1     

            # Report that the iteration limit was reached.
            if (step > cut_off and inpt_dict["verbose"] == 1):
                print("Pixel {}, {} hit iteration limit.".format(i,j))
    
    # Update data flags.
    segments["junidq"] = np.where(bad_pix_map != 0, 1, segments["junidq"])

    if (plot_step or save_step):
        # Create a plot of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        fig, ax = plt.subplots(figsize=(20,4))
        img_aspect, cbar_aspect = aspect_handler(bad_pix_alltime)
        im = ax.imshow(bad_pix_alltime/bad_pix_map.shape[0],origin='lower',cmap='viridis',
                       norm='linear',aspect=img_aspect,vmin=0,vmax=0.01)
        ax.set_title("Free-iteration DQ flags")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Fraction of integrations flagged')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-free_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create a plot of the maximum sigma difference that pixel felt.
        fig, ax = plt.subplots(figsize=(20,4))
        sig_diff = np.empty_like(sig_diffs[0,:,:])
        img_aspect, cbar_aspect = aspect_handler(sig_diff)
        for x1 in range(sig_diffs.shape[1]):
            for x2 in range(sig_diffs.shape[2]):
                sig_diff[x1,x2] = np.nanmax(sig_diffs[:,x1,x2])
        
        vmin, vmax = 0, 1.5*sigma
        im = ax.imshow(sig_diff,origin='lower',cmap='viridis',
                       norm='linear',aspect=img_aspect,vmin=vmin,vmax=vmax)
        ax.set_title("Maximum sigma difference from median")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Flux [DN]')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-free_biggest-differences.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Iterations complete. Total bad pixels found: %.0f."% np.count_nonzero(bad_pix_map))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return segments