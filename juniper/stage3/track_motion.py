import os
import time
from tqdm import tqdm

import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from astropy import modeling

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.stage4.align_spec import cross_correlate

def track_pos(segments, inpt_dict):
    """Tracks position of trace in each integration.

    Args:
        segments (xarray): its segments DataSet is the integrations which will
        be tracked.
        inpt_dict (dict): instructions for running this step.

    Returns:
        xarray, list, list, list, list: the segments array with updated data
        quality flags, and the disp. positions, cross-disp. positions, cross-disp.
        widths, and identified indices of bad frames.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Tracking motion of the trace...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Start tracking.
    bad_k = []
    bad_frame_map = np.zeros_like(segments.data)

    dispersion_position = []
    if inpt_dict["track_disp"]:
        dispersion_position = []
        # Need to make a template.
        collapsed = np.nansum(segments.data.values[:,:,:], axis=1) # collapse all frames on axis 1
        template = np.median(collapsed, axis=0) # take median in time
        template /= np.max(template) # normalise so peak is at 1
        template = medfilt(template, kernel_size=7)

        if (plot_step or save_step):
            # Create a plot in time of the dispersion profile used for cross-correlation.
            plt.figure(figsize=(5,5))
            plt.plot(template)
            plt.xlabel('Dispersion Position [pixels]')
            plt.ylabel('Normalised Flux [a.u.]')
            plt.tick_params(which='both',axis='both',direction='in')
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_dispersion_template.png"),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

        for k in tqdm(range(segments.data.shape[0]),
                    desc='Fitting trace dispersion position...',
                    disable=(not time_ints)):
            profile = np.nansum(segments.data.values[k,:,:], axis=0)
            profile = profile/np.max(profile) # normalize amplitude to 1 for ease of fit
            profile = medfilt(profile, kernel_size=7) # filter outliers to reduce their impact on the fit
            pos = fit_disp_profile(profile,template=template)
            dispersion_position.append(pos)

            # Plot an example..
            if (plot_step or save_step) and k == 0:
                # Create a plot of a dispersion correlation example, to show how the template is matched.
                plt.figure(figsize=(5,5))
                plt.plot(template, color='k',label='Dispersion Template')
                plt.plot(profile, color='red', ls='--', label='Frame To Correlate')
                plt.xlabel('Dispersion Position [pixels]')
                plt.ylabel('Normalised Flux [a.u.]')
                plt.tick_params(which='both',axis='both',direction='in')
                plt.legend()
                if save_step:
                    plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_dispersion_example.png"),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()
        
        if inpt_dict["reject_disp"]:
            # Flag any integration with sudden movement.
            med_disp, std_disp = np.median(dispersion_position), np.std(dispersion_position)

            for k in tqdm(range(segments.data.shape[0]),
                          desc='Identifying trace dispersion position outliers...',
                          disable=(not time_ints)):
                if np.abs(med_disp - dispersion_position[k]) > 3*std_disp:
                    # The frame moved by 3 sigma, kick it.
                    bad_k.append(k)
                    bad_frame_map[k,:,:] = np.ones_like(bad_frame_map[k,:,:]) # the whole frame is flagged for data quality
        
        # Plot the dispersion positions.
        if (plot_step or save_step):
            # Create a plot in time of the measured dispersion positions.
            plt.figure(figsize=(5,5))
            plt.scatter(segments.time.values, dispersion_position, color='k')
            if inpt_dict["reject_disp"]:
                # Plot lines marking where things were kicked.
                plt.axhline(med_disp,ls='--',color='red')
                for mult in (-1,1):
                    plt.axhline(med_disp+(mult*3*std_disp),ls=':',color='red')
            plt.xlabel('Exposure Time [MJD]')
            plt.ylabel('Dispersion Position [pixels]')
            plt.tick_params(which='both',axis='both',direction='in')
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_dispersion_positions.png"),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

    crossdispersion_position = []
    crossdispersion_width = []
    if inpt_dict["track_spatial"]:
        crossdispersion_position = []
        crossdispersion_width = []
        for k in tqdm(range(segments.data.shape[0]),
                    desc='Fitting trace cross-dispersion position and width...',
                    disable=(not time_ints)):
            profile = np.nansum(segments.data.values[k,:,:], axis=1)
            pos, width = fit_cdisp_profile(profile,guess_pos=profile.shape[0]*0.50,guess_width=1)
            crossdispersion_position.append(pos)
            crossdispersion_width.append(width)

        if inpt_dict["reject_spatial"]:
            # Flag any integration with sudden movement or blooming/defocusing.
            med_cross, std_cross = np.median(crossdispersion_position), np.std(crossdispersion_position)

            for k in tqdm(range(segments.data.shape[0]),
                          desc='Identifying trace cross-dispersion position outliers...',
                          disable=(not time_ints)):
                if np.abs(med_cross - crossdispersion_position[k]) > 3*std_cross:
                    # The frame moved by 3 sigma, kick it if it isn't already kicked.
                    if k not in bad_k:
                        bad_k.append(k)
                        bad_frame_map[k,:,:] = np.ones_like(bad_frame_map[k,:,:]) # the whole frame is flagged for data quality
        
        # Plot the cross-dispersion positions and widths.
        if (plot_step or save_step):
            # Create a plot in time of the measured cross-dispersion positions.
            plt.figure(figsize=(5,5))
            plt.scatter(segments.time.values, crossdispersion_position, color='k')
            if inpt_dict["reject_spatial"]:
                # Plot lines marking where things were kicked.
                plt.axhline(med_cross,ls='--',color='red')
                for mult in (-1,1):
                    plt.axhline(med_cross+(mult*3*std_cross),ls=':',color='red')
            plt.xlabel('Exposure Time [MJD]')
            plt.ylabel('Cross-Dispersion Position [pixels]')
            plt.tick_params(which='both',axis='both',direction='in')
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_cross-dispersion_positions.png"),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

            # Create a plot in time of the measured cross-dispersion widths.
            plt.figure(figsize=(5,5))
            plt.scatter(segments.time.values, crossdispersion_width, color='k')
            plt.xlabel('Exposure Time [MJD]')
            plt.ylabel('Cross-Dispersion Width [pixels]')
            plt.tick_params(which='both',axis='both',direction='in')
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_cross-dispersion_widths.png"),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

    # Update data flags.
    segments.dq.values = np.where(bad_frame_map != 0, 1, segments.dq.values)

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Frame tracking complete.")
        if bad_k:
            print("Total frames rejected for sudden motion: {}".format(len(bad_k)))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return segments, dispersion_position, crossdispersion_position, crossdispersion_width, bad_k

def fit_cdisp_profile(profile,guess_pos,guess_width):
    """Simple utility to fit a Gaussian profile to the trace cross-dispersion
    profile. Used to track position and width.

    Args:
        profile (np.array): a cross-dispersion profile whose position and width
        are to be tracked.
        guess_pos (float): initial guess for the position of the source.
        guess_width (float): initial guess for the width of the source.

    Returns:
        float, float: the position and sigma width of the profile.
    """
    #profile = profile/np.max(profile) # normalize amplitude to 1 for ease of fit
    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude=np.max(profile), mean=guess_pos, stddev=guess_width)
    fitted_model = fitter(model, [i for i in range(profile.shape[0])], profile)
    return fitted_model.mean[0], fitted_model.stddev[0]

def fit_disp_profile(profile, template):
    """Simple utility to cross-correlate a template profile to the trace dispersion
    profile. Used to track position.

    Args:
        profile (np.array): a dispersion profile whose position is to be tracked.
        template (np.array): a median dispersion profile used to look for
        dispersion-direction displacements.

    Returns:
        float: the position of the profile.
    """
    shift = cross_correlate(profile, template, tspc=50, hrf=0.005, tfit=90)
    return shift