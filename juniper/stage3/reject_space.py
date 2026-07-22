import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import aspect_handler

def smooth(segments, inpt_dict):
    """Uses median filtering to smooth outliers.

    Args:
        segments (dict): Its segments["data"] object is the data to remove outliers from.
        inpt_dict (dict): instructions for running this step.

    Returns:
        dict: segments with spatial outliers removed by median filtering.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Cleaning threshold=%.1f outliers with spatial filtering..." % inpt_dict["space_sigma"])

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found.
    bad_pix_map = np.zeros_like(segments["data"])
    bad_pix_removed = 0
    bad_pix_per_frame = []

    # Retain smoothed models and absolute differences for plotting.
    smooths = np.empty_like(segments["data"])
    abs_diffs = np.empty_like(segments["data"])

    # Iterate over each frame and smooth.
    for k in tqdm(range(segments["data"].shape[0]),
                  desc = 'Smoothing outliers from integrations...',
                  disable=(not time_ints)):
        # Build a smoothed model.
        smooth = median_filter(np.copy(segments["data"][k,:,:]),
                               size=inpt_dict["space_kernel"])
        smooths[k,:,:] = smooth
        
        # Compare data to smoothed model.
        abs_diff = np.abs(segments["data"][k,:,:]-smooth)
        abs_diffs[k,:,:] = abs_diff
        
        # Flag outliers on the bad pixel map.
        S = np.where(abs_diff > inpt_dict["space_sigma"]*np.nanmean(abs_diff),1,0)
        bad_pix_map[k,:,:] += S
        bad_pix_this_frame = np.count_nonzero(S)
        bad_pix_removed += bad_pix_this_frame

        # Count outliers.
        bad_pix_per_frame.append(bad_pix_this_frame)

        # And replace if asked.
        if inpt_dict["space_replace"]:
            segments["data"][k,:,:] = np.where(S == 1, smooth, segments["data"][k,:,:])
        # Otherwise, just mask the bad pixels.
        else:
            segments["badpixmask"][k,:,:] = np.where(bad_pix_map[k,:,:]>0,1,
                                                     segments["badpixmask"][k,:,:])
            
    # Update data flags.
    segments["junidq"] = np.where(bad_pix_map != 0, 1, segments["junidq"])

    # Report results of cleaning.
    if inpt_dict["verbose"] >= 1:
        print("Smoothing complete.")
        print("Median percentage of data cleaned by smoothing: {:.3f}%".format(100*np.median(bad_pix_per_frame)/(S.shape[0]*S.shape[1])))
    if inpt_dict["verbose"] == 2:
        print("Median outliers found in each frame: {} +/- {}".format(int(np.median(bad_pix_per_frame)),
                                                                      int(np.std((bad_pix_per_frame)))))
        print("Highest and lowest number of pixels treated in a frame: {:.0f} and {:.0f}".format(np.max(bad_pix_per_frame),
                                                                                                 np.min(bad_pix_per_frame)))

    if (plot_step or save_step):
        # Create plots of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        fig, ax = plt.subplots(figsize=(20,4))
        img_aspect, cbar_aspect = aspect_handler(bad_pix_alltime)
        im = ax.imshow(bad_pix_alltime/bad_pix_map.shape[0],origin='lower',cmap='viridis',
                       norm='linear',aspect=img_aspect,vmin=0,vmax=1)
        ax.set_title("Spatial smoothing DQ flags")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Fraction of integrations flagged')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-smoothing_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

        # Create a plot of the median smoothed model.
        fig, ax = plt.subplots(figsize=(20,4))
        medsmooth = np.median(smooths,axis=0)
        img_aspect, cbar_aspect = aspect_handler(medsmooth)
        vmin, vmax = np.nanpercentile(medsmooth,q=5), np.nanpercentile(medsmooth,q=95)
        if vmin <= 0:
            pos = medsmooth[medsmooth>0]
            vmin, vmax = np.nanpercentile(pos[np.isfinite(pos)],q=5), np.percentile(pos[np.isfinite(pos)],q=95)
        im = ax.imshow(medsmooth,origin='lower',cmap='viridis',
                       norm='log',aspect=img_aspect,vmin=vmin,vmax=vmax)
        ax.set_title("Median spatially-filtered model")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Flux [DN]')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-smoothing_med-model.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create a plot of the maximum absolute differences between the data and smoothed model.
        fig, ax = plt.subplots(figsize=(20,4))
        abs_diff = np.empty_like(abs_diffs[0,:,:])
        img_aspect, cbar_aspect = aspect_handler(abs_diff)
        for x1 in range(abs_diffs.shape[1]):
            for x2 in range(abs_diffs.shape[2]):
                abs_diff[x1,x2] = np.nanmax(abs_diffs[:,x1,x2])
        
        vmin, vmax = np.nanpercentile(abs_diff,q=5), np.nanpercentile(abs_diff,q=95)
        if vmin <= 0:
            pos = abs_diff[abs_diff>0]
            vmin, vmax = np.nanpercentile(pos[np.isfinite(pos)],q=5), np.percentile(pos[np.isfinite(pos)],q=95)
        im = ax.imshow(abs_diff,origin='lower',cmap='viridis',
                       norm='log',aspect=img_aspect,vmin=vmin,vmax=vmax)
        ax.set_title("Maximum difference between data and model")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Flux [DN]')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-smoothing_biggest-differences.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

    # Count outliers and log.
    if inpt_dict["verbose"] >= 1:
        print("Smoothing complete. Outliers found in total: {}".format(bad_pix_removed))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
        
    return segments


def led(segments, inpt_dict):
    """Convolves a Laplacian kernel with the obs.images to replace spatial outliers with
    the median of the surrounding 3x3 kernel.

    Args:
        segments (dict): Its segments["data"] object is the data to remove outliers from.
        inpt_dict (dict): instructions for running this step.

    Returns:
        dict: segments with spatial outliers removed by Laplacian Edge Detection.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Cleaning threshold=%.1f outliers with Laplacian edge detection..." % inpt_dict["led_sigma"])

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found.
    bad_pix_map = np.zeros_like(segments["data"])
    bad_pix_per_frame = []
    iterations_needed_per_frame = []

    # Retain noise models, fine structure models, laplacian images, and absolute differences for plotting.
    noise_models = np.empty_like(segments["data"])
    fine_structure_models = np.empty_like(segments["data"])
    contrast_images = np.empty_like(segments["data"])
    s_images = np.empty_like(segments["data"])

    # Define the Laplacian kernel.
    l = 0.25*np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

    # Iterate over each frame one at a time until the iteration stop condition is met by each frame.
    for k in tqdm(range(segments["data"].shape[0]),
                  desc = 'Running LED on integrations...',
                  disable=(not time_ints)):
        # Get the frame and errors as np.array objects so we can operate on them.
        integration = segments["data"][k]
        errs = segments["err"][k]

        # Track outliers flagged in this frame and iterations performed.
        bad_pix_removed = 0
        iteration_N = 1

        # Then start iterating over this frame and keep going until the iteration stop condition is met.
        stop_iterating = False
        while not stop_iterating:
            # Estimate readnoise value.
            var2 = errs**2 - integration
            var2[var2 < 0] = 0 # enforce positivity.
            rn = np.sqrt(var2) # estimate readnoise array

            # Build the noise model.
            noise_model = build_noise_model(integration, rn)
            if iteration_N == 1:
                noise_models[k,:,:] = noise_model
            if inpt_dict["fine_structure"]:
                F = build_fine_structure_model(integration)
                if iteration_N == 1:
                    fine_structure_models[k,:,:] = F

            # Subsample the array.
            subsample, original_shape = subsample_frame(integration,
                                                        factor=inpt_dict["led_factor"])
            
            # Convolve subsample with laplacian.
            lap_img = np.convolve(l.flatten(),subsample.flatten(),mode='same').reshape(subsample.shape)
            lap_img[lap_img < 0] = 0 # force positivity
            
            # Resample laplacian-convolved subsampled image to original size.
            resample = resample_frame(lap_img, original_shape)

            # Divide by the noise model scaled by the resampling factor.
            S = resample/(inpt_dict["led_factor"]*noise_model)
            
            # Remove sampling flux to protect data from being targeted by LED.
            S = S - median_filter(S, size=5)
            if iteration_N == 1:
                s_images[k,:,:] = S

            # Spot outliers.
            S[np.abs(S) < inpt_dict["led_sigma"]] = 0 # any not zero after this are rays.
            S[S!=0] = 1 # for visualization and comparison to fine structure model.

            # If we have a fine structure model, we also need to check the contrast.
            if inpt_dict["fine_structure"]:
                contrast_image = resample/F
                if iteration_N == 1:
                    contrast_images[k,:,:] = contrast_image
                contrast_image[contrast_image < inpt_dict["contrast_factor"]] = 0 # any not zero after this are rays.
                contrast_image[contrast_image!=0] = 1 # for visualization and comparison to sampling flux model.
                
                # Then we need to merge the results of S = Laplacian_image/factor*noise_model - sampling_flux
                # and contrast_image = Laplacian_image/Fine_structure_model so that we only take 1s where both are 1.
                S = np.where(S == contrast_image, S, 0)

            # Locate new bad pixels.
            bad_pix_last_frame = -100
            if iteration_N != 1:
                bad_pix_last_frame = bad_pix_this_frame # if this is not the first time we've done this, we need to store the last frame's bad pix count before updating it.

            S = np.where(bad_pix_map[k,:,:] == 1, 0, S) # if the pixel was already reported as bad in a previous iteration, don't double count it.
            bad_pix_map[k,:,:] += S
            bad_pix_this_frame = np.count_nonzero(S)
            bad_pix_removed += bad_pix_this_frame

            # Correct frames, if asked.
            if inpt_dict["led_replace"]:
                med_filter_image = median_filter(integration,size=5)
                integration = np.where(S != 0, med_filter_image, integration)
            # Otherwise, just mask the bad pixels.
            else:
                segments["badpixmask"][k,:,:] = np.where(bad_pix_map[k,:,:]>0,1,
                                                         segments["badpixmask"][k,:,:])

            # Increment iteration number and check if condition to stop iterating is hit.
            iteration_N += 1
            if (inpt_dict["led_n"] != None and iteration_N > inpt_dict["led_n"]): # if it has hit the iteration cap
                stop_iterating = True
            if (inpt_dict["led_n"] == None and bad_pix_this_frame == bad_pix_last_frame): # if it has stalled out on finding new outliers
                stop_iterating = True
        
        # Log cleaning info after the iterations for this frame have stopped.
        bad_pix_per_frame.append(bad_pix_removed)
        iterations_needed_per_frame.append(iteration_N-1)

        # And replace the xarray datasets if asked.
        if inpt_dict["led_replace"]:
            segments["data"][k] = np.where(segments["data"][k] != integration,integration,segments["data"][k])
    
    # Report results of cleaning.
    if inpt_dict["verbose"] >= 1:
        print("All LED iterations complete.")
        print("Median percentage of data cleaned by LED: {:.3f}%".format(100*np.median(bad_pix_per_frame)/(S.shape[0]*S.shape[1])))
    if inpt_dict["verbose"] == 2:
        print("Highest and lowest number of pixels treated in a frame: {:.0f} and {:.0f}".format(np.max(bad_pix_per_frame),
                                                                                                 np.min(bad_pix_per_frame)))
        print("Typical number of iterations needed: {:.0f}".format(np.median(iterations_needed_per_frame)))

    # Update data flags.
    segments["junidq"] = np.where(bad_pix_map != 0, 1, segments["junidq"])

    if (plot_step or save_step):
        # Create plots of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        fig, ax = plt.subplots(figsize=(20,4))
        img_aspect, cbar_aspect = aspect_handler(bad_pix_alltime)
        im = ax.imshow(bad_pix_alltime/bad_pix_map.shape[0],origin='lower',cmap='viridis',
                       norm='linear',aspect=img_aspect,vmin=0,vmax=1)
        ax.set_title("Laplacian Edge Detection DQ flags")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Fraction of integrations flagged')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

    if (plot_ints or save_ints):
        # Create plots of the median noise model, fine structure model, and S images.
        fig, ax = plt.subplots(figsize=(20,4))
        medimg = np.median(noise_models,axis=0)
        img_aspect, cbar_aspect = aspect_handler(medimg)
        vmin, vmax = np.nanpercentile(medimg,q=5), np.nanpercentile(medimg,q=95)
        if vmin <= 0:
            pos = medimg[medimg>0]
            vmin, vmax = np.nanpercentile(pos[np.isfinite(pos)],q=5), np.percentile(pos[np.isfinite(pos)],q=95)
        im = ax.imshow(medimg,origin='lower',cmap='viridis',
                       norm='log',aspect=img_aspect,vmin=vmin,vmax=vmax)
        ax.set_title("Median noise model")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Flux [DN]')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_median-noise-model.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(20,4))
        medimg = np.median(fine_structure_models,axis=0)
        img_aspect, cbar_aspect = aspect_handler(medimg)
        vmin, vmax = np.nanpercentile(medimg,q=5), np.nanpercentile(medimg,q=95)
        if vmin <= 0:
            pos = medimg[medimg>0]
            vmin, vmax = np.nanpercentile(pos[np.isfinite(pos)],q=5), np.percentile(pos[np.isfinite(pos)],q=95)
        im = ax.imshow(medimg,origin='lower',cmap='viridis',
                       norm='log',aspect=img_aspect,vmin=vmin,vmax=vmax)
        ax.set_title("Median fine structure model")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Flux [DN]')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_median-fine-structure-model.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(20,4))
        medimg = np.median(s_images,axis=0)
        img_aspect, cbar_aspect = aspect_handler(medimg)
        vmin, vmax = np.nanpercentile(medimg,q=5), np.nanpercentile(medimg,q=95)
        if vmin <= 0:
            pos = medimg[medimg>0]
            vmin, vmax = np.nanpercentile(pos[np.isfinite(pos)],q=5), np.percentile(pos[np.isfinite(pos)],q=95)
        im = ax.imshow(medimg,origin='lower',cmap='viridis',
                       norm='log',aspect=img_aspect,vmin=vmin,vmax=vmax)
        ax.set_title("Median S")
        cbar = plt.colorbar(mappable=im,orientation='horizontal',aspect=cbar_aspect)
        cbar.set_label('Flux [DN]')

        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_median-S.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("All integrations cleaned of spatial outliers by LED.")

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    
    return segments

def build_noise_model(data_frame, readnoise):
    """Builds a noise model for the given data frame, following van Dokkum 2001 methods.

    Args:
        data_frame (np.array): Integration from the segments["data"] array, used to build the noise model.
        readnoise (float): Readnoise estimated to be in the data frame.

    Returns:
        np.array: 2D array same size as the data frame, a noise model describing noise in the frame.
    """
    noise_model = np.sqrt(median_filter(np.abs(data_frame),size=5)+readnoise**2)
    noise_model[noise_model <= 0] = np.mean(noise_model) # really want to avoid nans
    return noise_model

def subsample_frame(data_frame, factor=2):
    """Subsamples the input frame by the given subsampling factor.

    Args:
        data_frame (np.array): Integration from the segments["data"] array.
        factor (int, optional): int >= 2. Factor by which to subsample the array. Defaults to 2.

    Returns:
        np.array: sub-sampled data frame.
    """
    factor = int(factor) # Force integer
    if factor < 2:
        print("Subsampling factor must be at least 2, forcing factor to 2...")
        factor = 2 # Force factor 2 or more
    
    original_shape = np.shape(data_frame)
    ss_shape = (original_shape[0]*factor,original_shape[1]*factor)
    subsample = np.empty(ss_shape)
    
    # Subsample the array.
    for i in range(ss_shape[0]):
        for j in range(ss_shape[1]):
            try:
                subsample[i,j] = data_frame[int((i+1)/2),int((j+1)/2)]
            except IndexError:
                subsample[i,j] = 0
    return subsample, original_shape

def resample_frame(data_frame, original_shape):
    """Resamples a subsampled array back to the original shape.

    Args:
        data_frame (np.array): Subsampled integration from the segments["data"] array.
        original_shape (tuple of int): Original shape of the subsampled array.

    Returns:
        np.array: 2D array with original shape resampled from the data frame.
    """
    resample = np.empty(original_shape)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            resample[i,j] = 0.25*(data_frame[2*i-1,2*j-1] +
                                  data_frame[2*i-1,2*j] +
                                  data_frame[2*i,2*j-1] +
                                  data_frame[2*i,2*j])
    return resample

def build_fine_structure_model(data_frame):
    """Builds a fine structure model for the data frame.

    Args:
        data_frame (np.array): Native resolution data.

    Returns:
        np.array: fine structure model built from the data.
    """
    F = median_filter(data_frame, size=3) - median_filter(median_filter(data_frame, size=3), size=7)
    F[F <= 0] = np.mean(F) # really want to avoid nans
    return F