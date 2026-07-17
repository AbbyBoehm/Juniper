import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.cleaning import get_com_mask

def extract(segments, inpt_dict):
    """Extract the 1D spectral flux using either standard (box) extraction
    or the optimum method of Horne 1986.

    Args:
        segments (xarray): its data DataSet contains the integrations to
        sum across, and its wavelengths DataSet can be used to limit the
        1D extraction to a certain wavelength range.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array, np.array, np.array: the one-dimensional spectrum and
        corresponding wavelengths and errors for that spectrum.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Extracting 1D spectrum with {} method...".format(inpt_dict["extract_method"]))

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Retain masks for plotting.
    extraction_masks = []

    # Initialize arrays.
    oneD_spec = np.empty((segments["data"].shape[0],segments["data"].shape[2])) # it has shape nints x ncols
    oneD_err = np.empty((segments["data"].shape[0],segments["data"].shape[2])) # same shape as oneD_spec
    wav_sols = np.empty((segments["data"].shape[0],segments["data"].shape[2])) # same shape as oneD_spec

    # Build profile, if applicable.
    profiles = np.ones_like(segments["data"]) # neutral weights, if not optimizing.
    if inpt_dict["extract_method"] == "optimum":
        # Now we have to actually build real profiles.
        if inpt_dict["aperture_type"] == "median":
            profile = optimum_median(segments)
        if inpt_dict["aperture_type"] == "custom":
            profile = np.load(inpt_dict["aperture_path"],allow_pickle=True)
        for i in tqdm(range(profiles.shape[0]),
                      desc='Building optimum profiles for each frame...',
                      disable=(not time_ints)):
            profiles[i,:,:] = profile
    # Build com_mask, if applicable.
    if inpt_dict["com_halfwidth"]:
        # Build aperture based on where com is.
        com_mask = get_com_mask(np.nanmedian(segments["data"], axis=0),
                                width=inpt_dict["com_halfwidth"])
        # Invert com_mask so that data is unmasked while background is masked
        com_mask = np.where(com_mask == 1, 0, 1)

    # And populate.
    for i in tqdm(range(oneD_spec.shape[0]),
                  desc = 'Extracting spectra from each integration...',
                  disable=(not time_ints)):
        # Get the integration, wavelengths, and data quality.
        data_i = segments["data"][i,:,:]
        error_i = segments["err"][i,:,:]
        wav_i = segments["wavelengths"][i,:,:]

        if inpt_dict["columns"]:
            # Remove columns outside of the approved range via wavelength solution.
            l,r = inpt_dict["columns"]
            wav_i[:,:l] = 0
            wav_i[:,r:] = 0

        # Start building a mask.
        mask = np.zeros_like(data_i)

        if inpt_dict["wavelengths"]:
            # Mask wavelengths that are too short.
            mask = np.where(wav_i < inpt_dict["wavelengths"][0],1,mask)
            # And wavelengths that are too long.
            mask = np.where(wav_i > inpt_dict["wavelengths"][1],1,mask)

        if inpt_dict["mask_bad_pix"]:
            # Mask pixels using the bad pix mask.
            mask = np.where(segments["badpixmask"] != 0, 1, mask)
        
        # Mask where is outside of the aperture.
        if inpt_dict["com_halfwidth"]:
            # Apply com_mask to data.
            mask = np.where(com_mask == 1, 1, mask)
        else:
            # Aperture is defined by two bounding rows chosen by user.
            lower, upper = inpt_dict["aperture"]
            mask[0:lower] = 1
            mask[upper:] = 1

        # Finally, apply mask to anywhere the wavelength solution or data are off.
        mask = np.where(np.isnan(wav_i),1,mask) # if the wavelength solution is nan, mask the pixel
        mask = np.where(wav_i==0,1,mask) # also mask where the wavelength solution is 0 nm, that shouldn't happen
        mask = np.where(np.isnan(data_i),1,mask) # and mask any nans in the data itself
        #mask = np.where(np.isnan(e),1,mask) # and mask any nans in the errors too

        extraction_masks.append(mask)

        # Now apply the mask to the data and sum it on columns.
        data_i = np.ma.masked_array(data_i, mask=mask)
        oneD_spec[i,:] = np.ma.sum(data_i,axis=0)

        # Apply the mask to the errors, which add in quadrature.
        error_i = np.ma.masked_array(error_i, mask=mask)
        oneD_err[i,:] = np.sqrt(np.nansum(np.square(error_i),axis=0))

        # Apply the mask to the wavelengths and median it on columns.
        wav_i = np.ma.masked_array(wav_i, mask=mask)
        wav_sols[i,:] = np.ma.median(wav_i,axis=0)

        # If we are doing optimum, we must revise our extraction.
        if inpt_dict["extract_method"] == "optimum":
            # Need to renormalize the profiles to exclude masked regions.
            profiles[i,:,:] = np.where(mask==1,0,profiles[i,:,:])
            profiles[i,:,:] = profiles[i,:,:]/np.nansum(profiles[i,:,:], axis=0)

            # Need to revise the variance estimates using the standard box spectrum.
            variance = error_i**2 - data_i
            variance[variance<=0] = 1e-10
            standard_spectrum = oneD_spec[i,:]
            revised_variance = variance+np.abs(standard_spectrum[np.newaxis,:]*profiles[i,:,:])

            # Then weight the data and errors.
            optimized_data = (profile*data_i/revised_variance)/np.sum(profiles[i,:,:]**2 / revised_variance, axis=0)
            optimized_errors = (profile*error_i/revised_variance)/np.sum(profiles[i,:,:]**2 / revised_variance, axis=0)

            # Now revise oneD_spec and err.
            data_i = np.ma.masked_array(optimized_data, mask=mask)
            oneD_spec[i,:] = np.ma.sum(data_i,axis=0)

            error_i = np.ma.masked_array(optimized_errors, mask=mask)
            oneD_err[i,:] = np.sqrt(np.ma.sum(np.square(error_i),axis=0))

    if (plot_step or save_step):
        plt.imshow(np.median(np.array(extraction_masks),axis=0),
                   aspect=5,origin='lower')
        plt.title('Median 1D extraction mask')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_mask_median.png'),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

        if inpt_dict["extract_method"] == "optimum":
            plt.imshow(np.median(profiles,axis=0), vmin=0, vmax=1,
                       aspect=5,origin='lower')
            plt.title('1D extraction median optimum profile')
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_profile_median.png'),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

    # Delete any wavelength indices that have wavelength 0.
    wav_ref = wav_sols[0,:] # time[0] x wavelength, across time all wav sols should be same
    bad_indices = np.argwhere(wav_ref == 0)
    oneD_spec = np.delete(oneD_spec, bad_indices, axis=1)
    oneD_err = np.delete(oneD_err, bad_indices, axis=1)
    wav_sols = np.delete(wav_sols, bad_indices, axis=1)

    if (plot_step or save_step):
        fig,ax = plt.subplots(figsize = (7,5))
        ax.scatter(segments["time"],np.sum(oneD_spec,axis=1),color='darkblue')
        ax.set_title('Broad-band light curve')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_broadband-initial.png'),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    
    return oneD_spec, oneD_err, wav_sols

def optimum_median(segments):
    """Builds the spatial optimum profile using the median data frame.

    Args:
        segments (xarray): its data DataSet contains the integrations to
        build the profile with.

    Returns:
        np.array: the profiles array.
    """
    # Take the median of the segments on time.
    median_frame = np.nanmedian(segments["data"], axis=0)
    # Force positivity.
    median_frame[median_frame < 0] = 0
    # And normalize.
    median_frame = median_frame/np.nansum(median_frame, axis=0)
    return median_frame