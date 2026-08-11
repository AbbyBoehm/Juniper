import os
import time
from tqdm import tqdm
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from photutils.centroids import centroid_com
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def extract(segments, inpt_dict):
    """Extract the photometric flux using either standard (unweighted) extraction
    or the optimum method of Horne 1986.

    Args:
        segments (xarray): its data DataSet contains the integrations to sum across.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array, np.array: the photometric time-series and errors.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Extracting photometric time-series with {} method...".format(inpt_dict["extract_method"]))

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
    signal_tseries = np.empty((segments["data"].shape[0],)) # it has shape nints
    signal_err = np.empty((segments["data"].shape[0],)) # same shape as signal_tseries

    # Check for x-y positions from S3. If these aren't available, get a median x-y.
    if np.std(segments["cdisp"]) == np.std(segments["disp"]) == 0:
         # A fixed x-y at 0 only happens if you did not track in S3.
         if inpt_dict["verbose"] == 2:
             print("Stage 3 x-y solutions not found; will use median x-y position for all integrations.")
         median_image = np.median(segments["data"],axis=0)
         x, y = centroid_com(median_image)
         segments["disp"][:] = x
         segments["cdisp"][:] = y

    # Optimize aperture and annulus radii if asked.
    aperture_radius, inner_annulus_radius, outer_annulus_radius = [inpt_dict[key] for key in ("ap_radius","bckg_inner_ap","bckg_outer_ap")]
    if any([inpt_dict[key] == "auto" for key in ("ap_radius","bckg_inner_ap","bckg_outer_ap")]):
        # Get the out-of-event indices and relative time.
        idxstart, idxend = inpt_dict["idx_outofevent"]
        rel_time = segments["time"][idxstart:idxend] - segments["time"][0]
        # Any that aren't auto are held fixed; the others are varied until the optimum is found.
        ap_radius, bckg_inner_ap, bckg_outer_ap = [[inpt_dict[key],] for key in ("ap_radius","bckg_inner_ap","bckg_outer_ap")]
        if inpt_dict["ap_radius"] == "auto":
            ap_radius = [k for k in range(inpt_dict["ap_rad_limits"][0],inpt_dict["ap_rad_limits"][1]+1)]
        if inpt_dict["bckg_inner_ap"] == "auto":
            bckg_inner_ap = [k for k in range(inpt_dict["bc_inn_limits"][0],inpt_dict["bc_inn_limits"][1]+1)]
        if inpt_dict["bckg_outer_ap"] == "auto":
            bckg_outer_ap = [k for k in range(inpt_dict["bo_out_limits"][0],inpt_dict["bo_out_limits"][1]+1)]
        # We have to test every combination of ap_radius, bckg_inner_ap, and bckg_outer_ap that makes sense.
        ap_fluxes = []
        for ap_r in tqdm(ap_radius,desc='Extracting aperture flux to optimize aperture radius...',
                         disable=(not time_ints and len(ap_radius)==1)):
            ap_flux = []
            # Extract only from the out-of-event indices.
            for k in range(idxstart,idxend):
                # Define the aperture.
                aperture = CircularAperture((segments["disp"][k],segments["cdisp"][k]),r=ap_r)
                # Extract the out-of-event flux.
                flx = aperture_photometry(segments["data"][k,:,:],aperture)
                ap_flux.append(flx[0][-1])
            # And get the with-background aperture flux.
            ap_fluxes.append([ap_r,np.array(ap_flux)])
        
        # Now get the annuli fluxes.
        an_fluxes = []
        annuli = [x for x in product(bckg_inner_ap,bckg_outer_ap) if x[0] < x[1]]
        for an in tqdm(annuli,desc='Extracting annulus flux to optimize annulus radii...',
                       disable=(not time_ints and len(annuli)==1)):
            an_flux = []
            # Extract only from the out-of-event indices.
            for k in range(idxstart,idxend):
                # Define the aperture.
                aperture = CircularAnnulus((segments["disp"][k],segments["cdisp"][k]),an[0],an[1])
                # Extract the out-of-event flux.
                flx = aperture_photometry(segments["data"][k,:,:],aperture)
                an_flux.append(flx[0][-1])
            # And get the with-background aperture flux.
            an_fluxes.append([an[0],an[1],np.array(an_flux)])
        
        # Now optimize combinations of ap and an flux for the one that minimizes scatter.
        scatters = []
        # Take only combinations where the aperture radius is inside the inner annulus radius.
        aper_annuli = [x for x in product(ap_fluxes,an_fluxes) if x[0][0]<x[1][0]]
        for ap_an in tqdm(aper_annuli,desc='Optimizing over annulus radii...',
                          disable=(not time_ints and len(aper_annuli)==1)):
            # Get the fluxes.
            aper_flux, annulus_flux = ap_an[0][1], ap_an[1][2]
            # Get the radii.
            ap_r, inner, outer = ap_an[0][0], ap_an[1][0], ap_an[1][1]
            # Scale the annulus flux by the area.
            aper_area, annulus_area = np.pi*ap_r**2, np.pi*(outer**2-inner**2)
            scaled_annulus_flux = (aper_area/annulus_area)*annulus_flux
            # De-background the aperture flux.
            aper_flux -= scaled_annulus_flux
            # Linear fit to get scatter.
            norm_aper_flux = aper_flux/np.nanmedian(aper_flux)
            p = np.polyfit(rel_time,norm_aper_flux,1)
            model = np.polyval(p,rel_time)
            res = np.sum((model-norm_aper_flux)**2)
            scatters.append([res,ap_r,inner,outer])
        
        # Sort scatters by minimized scatter. 0th key will now be the optimized setings.
        scatters = sorted(scatters, key = lambda x:x[0])
        aperture_radius, inner_annulus_radius, outer_annulus_radius = scatters[0][1:]
        if inpt_dict["verbose"] == 2:
            print("Optimized extraction aperture and annulus to:",scatters[0][1:])

    if (plot_step or save_step):
        idxstart, idxend = (0,int(0.2*len(segments["disp"])))
        x0, y0 = np.median(segments["disp"][idxstart:idxend]), np.median(segments["cdisp"][idxstart:idxend])
        # Plot the optimized apertures over the median out-of-event frame.
        lin_threshold = 0.1
        vmin, vmax = np.nanpercentile(segments["data"][idxstart:idxend,:,:],q=1), np.nanpercentile(segments["data"][idxstart:idxend,:,:],q=99)
        symlog_norm_bckgs = colors.SymLogNorm(linthresh=lin_threshold, 
                                        linscale=1, 
                                        vmin=vmin,
                                        vmax=vmax,
                                        base=10)
        fig, ax = plt.subplots(figsize=(5,5))
        medimg = np.median(segments["data"][idxstart:idxend,:,:],axis=0)
        im = plt.imshow(medimg,cmap='viridis',origin='lower',
                        norm=symlog_norm_bckgs,aspect=1)
        
        # Draw three circles for the aperture and annulus.
        aper_circle = Circle((x0,y0),aperture_radius,edgecolor='white',facecolor='none',ls='-')
        ax.add_patch(aper_circle)

        inner_circle = Circle((x0,y0),inner_annulus_radius,edgecolor='white',facecolor='none',ls='--')
        ax.add_patch(inner_circle)

        outer_circle = Circle((x0,y0),outer_annulus_radius,edgecolor='white',facecolor='none',ls='--')
        ax.add_patch(outer_circle)
        
        cbar = plt.colorbar(mappable=im,orientation='vertical')
        cbar.set_label("Flux [DN]")
        ax.set_title("Aperture and annulus for photometry")
        ax.scatter(x0,y0,color='k',
                    marker='x',s=20,label='PSF COM')
        ax.legend()
        ax.set_xlabel('Detector X Axis')
        ax.set_ylabel('Detector Y Axis')
        ax.tick_params(which='both',axis='both',direction='in')
        if save_step:
            plt.savefig(os.path.join(inpt_dict["plot_dir"],"S5_extraction-aperture.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

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

    # And populate.
    for i in tqdm(range(signal_tseries.shape[0]),
                  desc = 'Extracting flux from each integration...',
                  disable=(not time_ints)):
        # Get the integration and data quality.
        data_i = segments["data"][i,:,:]
        error_i = segments["err"][i,:,:]

        # Start building a mask.
        mask = np.zeros_like(data_i)

        if inpt_dict["mask_bad_pix"]:
            # Mask pixels using the bad pix mask.
            mask = np.where(segments["badpixmask"] != 0, 1, mask)

        # Finally, apply mask to anywhere the wavelength solution or data are off.
        mask = np.where(np.isnan(data_i),1,mask) # and mask any nans in the data itself

        extraction_masks.append(mask)

        # Define the circle and annulus apertures and areas.
        aperture = CircularAperture((segments["disp"][i],segments["cdisp"][i]),aperture_radius)
        annulus = CircularAnnulus((segments["disp"][i],segments["cdisp"][i]),inner_annulus_radius, outer_annulus_radius)
        aper_area, annulus_area = np.pi*aperture_radius**2, np.pi*(outer_annulus_radius**2-inner_annulus_radius**2)

        # Now apply the mask to the data.
        data_i = np.ma.masked_array(data_i, mask=mask)
        aper_flux = aperture_photometry(data_i,aperture)[0][-1]
        # Get the background flux.
        annulus_flux = aperture_photometry(data_i,annulus)[0][-1]
        # Scale, correct, and collect!
        signal_tseries[i] = aper_flux-((aper_area/annulus_area)*annulus_flux)

        # Apply the mask to the errors, which add in quadrature.
        error_i = np.ma.masked_array(error_i, mask=mask)
        signal_err[i] = np.sqrt(aperture_photometry(error_i**2,annulus)[0][-1])

        # If we are doing optimum, we must revise our extraction.
        if inpt_dict["extract_method"] == "optimum":
            # Need to renormalize the profiles to exclude masked regions.
            profiles[i,:,:] = np.where(mask==1,0,profiles[i,:,:])
            profiles[i,:,:] = profiles[i,:,:]/np.nansum(profiles[i,:,:], axis=0)

            # Need to revise the variance estimates using the standard box spectrum.
            variance = error_i**2 - data_i
            variance[variance<=0] = 1e-10
            standard_spectrum = signal_tseries[i]
            revised_variance = variance+np.abs(standard_spectrum[np.newaxis,:]*profiles[i,:,:])

            # Then weight the data and errors.
            optimized_data = (profile*data_i/revised_variance)/np.sum(profiles[i,:,:]**2 / revised_variance, axis=0)
            optimized_errors = (profile*error_i/revised_variance)/np.sum(profiles[i,:,:]**2 / revised_variance, axis=0)

            # Now revise signal_tseries and err.
            data_i = np.ma.masked_array(optimized_data, mask=mask)
            signal_tseries[i] = np.ma.sum(data_i,axis=0)

            error_i = np.ma.masked_array(optimized_errors, mask=mask)
            signal_err[i] = np.sqrt(np.ma.sum(np.square(error_i),axis=0))

    if (plot_step or save_step):
        plt.imshow(np.median(np.array(extraction_masks),axis=0),
                   aspect=1,origin='lower')
        plt.title('Median photometric extraction mask')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_photometric_extraction_mask_median.png'),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

        if inpt_dict["extract_method"] == "optimum":
            plt.imshow(np.median(profiles,axis=0), vmin=0, vmax=1,
                       aspect=1,origin='lower')
            plt.title('1D extraction median optimum profile')
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_photometric_extraction_profile_median.png'),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

    if (plot_step or save_step):
        fig,ax = plt.subplots(figsize = (7,5))
        ax.scatter(segments["time"],signal_tseries,color='darkblue')
        ax.set_title('Photometric light curve')
        if save_step:
            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_photometric_extraction_broadband-initial.png'),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    
    return signal_tseries, signal_err

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