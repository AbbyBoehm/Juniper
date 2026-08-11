import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy import signal, ndimage
from scipy.optimize import least_squares

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import plot_allan

def bin_light_curves(spectra, inpt_dict):
    """Bins 1D spectra to return a light curve.

    Args:
        spectra (dict): its "spectrum" key has shape [detectors, time,
        wavelength], and its "waves" and "time" keys will be used in binning.
        inpt_dict (dict): instructions for running this step.

    Returns:
        dict: light_curves dictionary.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Binning 1D spectra to produce broad-band and spectroscopic light curves...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])
    
    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Initialize some lists at the detector level.
    broadband = [] # shape detector x time
    broaderr = [] # shape detector x time
    broadwave = [] # shape detector
    broadbins = [] # shape detector
    spec = [] # shape detector x time x central_wavelength
    specerr = [] # shape detector x time x central_wavelength
    specwave = []  # shape detector x central_wavelength
    specbins = [] # shape detector x wavelength edges

    # And begin processing detectors.
    for d in tqdm(range(len(spectra["spectrum"])),#.spectrum.shape[0]),
                  desc='Extracting light curves from each detector...',
                  disable=(not time_ints)):
        # Load the spectra and uncertainties and wavelength solutions from this detector.
        spectrum = spectra["spectrum"][d]
        error = spectra["errors"][d]
        waves = np.median(spectra["waves"][d],axis=0)

        if plot_step or save_step:
            # Plot the normalized 1D spec in time to show the transit.
            plt.imshow(np.copy(spectrum)/np.median(spectrum,axis=0),cmap='viridis',vmin=0.999,vmax=1.001)
            plt.colorbar(label="normalized flux")
            plt.xticks(ticks=list(range(0,len(waves),200)),
                       labels=[np.round(i,2) for i in waves[list(range(0,len(waves),200))]])
            plt.xlabel('wavelength [um]')
            plt.ylabel('time [mjd]')
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_1Dspec.png'.format(d)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

        # Set up a mask for NaN, inf, zero flux, zero-waves, high-error, and bad columns.
        column_mask = np.where(np.isnan(spectrum),1,0) # create a nan mask
        column_mask = np.where(np.isinf(spectrum),1,column_mask) # mask anywhere the spectrum is inf too
        column_mask = np.where(spectrum==0,1,column_mask) # and mask anywhere the spectrum is zero
        column_mask = np.where(error>np.median(error)+10*np.std(error),1,column_mask) # ignore freakishly large errors
        column_mask = np.where(waves==0,1,column_mask) # and ignore places with no wavelengths

        if inpt_dict["verbose"] == 2:
            ntime, ncols = spectrum.shape[0],spectrum.shape[1]
            print(f"For detector {d} with {ncols} columns, the following bad columns were recognized:")
            print("NaN spectral columns:",len(np.argwhere(np.isnan(spectrum)))/ntime)
            print("Infinite spectral columns:",len(np.argwhere(np.isinf(spectrum)))/ntime)
            print("Empty spectral columns:",len(np.argwhere(spectrum==0))/ntime)
            print("10-sigma abnormalities in error:",round(len(np.argwhere(error>np.median(error)+10*np.std(error)))/ntime))
            print("Invalid wavelength solutions:",len(np.argwhere(waves==0))/ntime)

        # Initialize detector-specific lists.
        spec_det = [] # will have shape time x central_wavelength
        specerr_det = []
        specwave_det = [] # will have shape central_wavelength
        specbins_det = [] # will have shape wavelength edges

        # Check for bad columns.
        if inpt_dict["reject_bad_cols"]:
            # Median normalize the 1D spectra over time.
            norm_spec = np.copy(spectrum)/np.median(spectrum,axis=0)
            norm_std = np.std(norm_spec,axis=0)
            compare = signal.medfilt(norm_std,21)
            std_check = np.abs(norm_std - compare)

            # Anywhere that std_check is above the threshold, we kick that column.
            bad_columns = np.argwhere(std_check>inpt_dict["bad_col_thres"])
            column_mask[:,bad_columns] = 1 # update the mask

            if inpt_dict["verbose"] == 2:
                print("Abnormally high standard deviations on flux (bad cols):",len(bad_columns))

            if (plot_ints or save_ints):
                # Create diagnostic plot of how many columns were marked as bad, and what the column std was.
                plt.plot(norm_std, color='k')
                plt.plot(compare, ls='--', color='red')
                plt.title("Standard deviation by column")
                plt.xlabel("column #")
                plt.ylabel("std dev")
                if save_ints:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_standarddeviation.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show(block=True)
                plt.close()

                plt.plot(std_check)
                plt.axhline(inpt_dict["bad_col_thres"], ls='--', color='red')
                plt.title("Measured - expected standard deviation")
                plt.xlabel("column #")
                plt.ylabel("abs(median filtered - true std dev)")
                plt.ylim(0,0.2)
                if save_ints:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_kickedcolumns.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show(block=True)
                plt.close()
        
        # Apply mask whether it is or is not empty.
        spectrum = np.ma.masked_array(spectrum,mask=column_mask)
        error = np.ma.masked_array(error,mask=column_mask)
        waves = np.ma.masked_array(waves,mask=np.median(column_mask,axis=0))

        # If asked, truncate the broadband
        broadband_spectrum = np.copy(spectrum)
        broadband_error = np.copy(error)
        broadband_waves = np.copy(waves)
        if inpt_dict["broadband_waves"][d]:
            waves_mask = np.where(waves < inpt_dict["broadband_waves"][d][0],-1,waves) # mask waves that are below our lower limit
            waves_mask = np.where(waves_mask > inpt_dict["broadband_waves"][d][1],-1,waves_mask) # mask waves above the upper limit
            waves_mask = np.where(waves_mask == -1,1,0) # the mask is 1 where things were bad, and 0 otherwise.
            mask=np.copy(column_mask)
            for i in range(mask.shape[0]):
                mask[i,:] += waves_mask
                mask[i,:] = np.where(mask[i,:] != 0,1,0)
            # Apply the new waves + bad pixels mask to the data.
            broadband_spectrum = np.ma.masked_array(spectrum,mask=mask)
            broadband_error = np.ma.masked_array(error,mask=mask)
            broadband_waves = np.ma.masked_array(waves,mask=np.median(mask,axis=0))

        # The broad-band light curve is trivial.
        broadband_det = np.ma.sum(broadband_spectrum,axis=1) # sum on wavelengths to get flux over whole bandpass
        # Broad-band uncertainties sum in quadrature.
        broaderr_det = np.ma.sqrt(np.ma.sum(broadband_error**2,axis=1)) # sum on wavelengths to get error over whole bandpass
        # Getting the central wavelength is simple.
        broadwave_det = np.ma.median(broadband_waves)
        # The wavelength bounds is also straightforward.
        broadbins_det = np.array([np.ma.min(broadband_waves),np.ma.max(broadband_waves)])
        # If asked, replace outliers from the curve.
        if inpt_dict["clip_outliers"]:
            broadband_det, n_changed = running_median_filter(broadband_det,
                                                             sigma=inpt_dict["clip_outliers"],
                                                             window=inpt_dict["clip_window"])
            if inpt_dict["verbose"] == 2:
                print("Replaced {} outliers in the broadband detector light curve.".format(n_changed))
        # Store both the spectrum and each point's uncertainty.
        broadband.append(broadband_det)
        broaderr.append(broaderr_det)
        broadwave.append(broadwave_det)
        broadbins.append(broadbins_det)

        if (plot_step or save_step):
            # Create diagnostic plot of the broad-band light curve.
            plt.errorbar(spectra["time"][d], broadband_det, yerr=broaderr_det, fmt='ko', capsize=3)
            plt.title("Broad-band light curve")
            plt.xlabel("time [mjd]")
            plt.ylabel("flux [a.u.]")
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_broadband_lc.png'.format(d)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

            # Also, create Allan Variance plot.
            if (plot_ints or save_ints):
                fig, ax = allan_variance(spectra["time"][d], broadband_det)
                if save_ints:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_Allanbroadband_lc.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show(block=True)
                plt.close()

        # The spec curves are more nuanced.
        if inpt_dict["bin_method"] == "columns":
            # Bin every n_columns columns.
            n_columns = inpt_dict["n_columns"][d]
            l = 0
            r = n_columns

            # Get expected number of bins.
            expected = int(spectrum.shape[1]/n_columns)
            pbar = tqdm(total = expected,
                        desc='Binning spectroscopic light curves by columns...',
                        disable=(not time_ints))
            while r < spectrum.shape[1]:
                # Get the light curve and uncertainties for this bin.
                bin_spec = np.ma.sum(spectrum[:,l:r],axis=1)
                bin_err = np.ma.sqrt(np.ma.sum(np.square(error[:,l:r]),axis=1))
                bin_wave = np.ma.median(waves[l:r])
                bin_bins = np.array([np.ma.min(waves[l:r]),np.ma.max(waves[l:r])])
                if bin_bins[0] == bin_bins[1]:
                    # Should only happen if len(wavs[l:r]) == 1.
                    try:
                        next_wave = np.ma.median(waves[r:r+n_columns])
                        hw = (next_wave-bin_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])
                    except:
                        last_wave = specwave_det[-1]
                        hw = (bin_wave-last_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])

                # If asked, replace outliers from the curve.
                if inpt_dict["clip_outliers"]:
                    bin_spec, n_changed = running_median_filter(bin_spec,
                                                             sigma=inpt_dict["clip_outliers"],
                                                             window=inpt_dict["clip_window"])
                    if inpt_dict["verbose"] == 2:
                        print("Replaced {} outliers in the spectroscopic light curve at {:.3F} um.".format(n_changed, bin_wave))
        
                # Store both the spectrum and each point's uncertainty.
                spec_det.append(bin_spec)
                specerr_det.append(bin_err)
                specwave_det.append(bin_wave)
                specbins_det.append(bin_bins)

                if (plot_step or save_step):
                    # Create diagnostic plot of this spec light curve.
                    plt.errorbar(spectra["time"][d], bin_spec, yerr=bin_err, fmt='ko', capsize=3)
                    plt.title("Spectroscopic light curve")
                    plt.xlabel("time [mjd]")
                    plt.ylabel("flux [a.u.]")
                    if save_step:
                        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_spec{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                    dpi=300, bbox_inches='tight')
                    if plot_step:
                        plt.show(block=True)
                    plt.close()

                    if (plot_ints or save_ints):
                        # Also, create Allan Variance plot.
                        fig, ax = allan_variance(spectra["time"][d], bin_spec)
                        if save_ints:
                            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_Allan{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                        dpi=300, bbox_inches='tight')
                        if plot_ints:
                            plt.show(block=True)
                        plt.close()

                # Progress bar update.
                pbar.update(1)

                # Advance bins.
                l = r
                r += n_columns
            # Add the final bin.
            bin_spec = np.ma.sum(spectrum[:,l:],axis=1)
            bin_err = np.ma.sqrt(np.ma.sum(np.square(error[:,l:]),axis=1))
            bin_wave = np.ma.median(waves[l:])
            bin_bins = np.array([np.ma.min(waves[l:]),np.ma.max(waves[l:])])
            if bin_bins[0] == bin_bins[1]:
                last_wave = specwave_det[-1]
                hw = (bin_wave-last_wave)/2
                bin_bins=np.array([bin_wave-hw,bin_wave+hw])

            # If asked, replace outliers from the curve.
            if inpt_dict["clip_outliers"]:
                bin_spec, n_changed = running_median_filter(bin_spec,
                                                            sigma=inpt_dict["clip_outliers"],
                                                            window=inpt_dict["clip_window"])
                if inpt_dict["verbose"] == 2:
                    print("Replaced {} outliers in the spectroscopic light curve at {:.3F} um.".format(n_changed, bin_wave))
    
            # Store both the spectrum and each point's uncertainty.
            spec_det.append(bin_spec)
            specerr_det.append(bin_err)
            specwave_det.append(bin_wave)
            specbins_det.append(bin_bins)

            if (plot_step or save_step):
                # Create diagnostic plot of this spec light curve.
                plt.errorbar(spectra["time"][d], bin_spec, yerr=bin_err, fmt='ko', capsize=3)
                plt.title("Spectroscopic light curve")
                plt.xlabel("time [mjd]")
                plt.ylabel("flux [a.u.]")
                if save_step:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_spec{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()

                if (plot_ints or save_ints):
                    # Also, create Allan Variance plot.
                    fig, ax = allan_variance(spectra["time"][d], bin_spec)
                    if save_ints:
                        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_Allan{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()

            # Close the progress bar.
            pbar.close()
        
        elif inpt_dict["bin_method"] == "wavelengths":
            # Bin parts of 1D spec that are in between each wavelength bin.
            wave_bins = inpt_dict["wave_bins"][d]

            for i in tqdm(range(1,len(wave_bins)),
                          desc='Binning spectroscopic light curves by wavelength...',
                          disable=(not time_ints)):
                wl, wr = wave_bins[i-1],wave_bins[i] # defines edges of this bin
                wave_bin = np.array([k[0] for k in np.argwhere(((waves >= wl) & (waves <= wr)))])

                # Get the light curve and uncertainties for this bin.
                bin_spec = np.ma.sum(spectrum[:,wave_bin],axis=1)
                bin_err = np.ma.sqrt(np.ma.sum(np.square(error[:,wave_bin]),axis=1))
                bin_wave = np.ma.median(waves[wave_bin])
                bin_bins = np.array([np.ma.min(waves[wave_bin]),np.ma.max(waves[wave_bin])])
                if bin_bins[0] == bin_bins[1]:
                    # Should only happen if len(wavs[l:r]) == 1.
                    try:
                        next_wave_bin = np.argwhere(((waves >= wave_bins[i]) & (waves <= wave_bins[i+1])))
                        next_wave = np.ma.median(waves[next_wave_bin])
                        hw = (next_wave-bin_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])
                    except:
                        last_wave = specwave_det[-1]
                        hw = (bin_wave-last_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])

                # If asked, replace outliers from the curve.
                if inpt_dict["clip_outliers"]:
                    bin_spec, n_changed = running_median_filter(bin_spec,
                                                             sigma=inpt_dict["clip_outliers"],
                                                             window=inpt_dict["clip_window"])
                    if inpt_dict["verbose"] == 2:
                        print("Replaced {} outliers in the spectroscopic light curve at {:.3F} um.".format(n_changed, bin_wave))
        
                # Store both the spectrum and each point's uncertainty.
                spec_det.append(bin_spec)
                specerr_det.append(bin_err)
                specwave_det.append(bin_wave)
                specbins_det.append(bin_bins)

                if (plot_step or save_step):
                    # Create diagnostic plot of this spec light curve.
                    plt.errorbar(spectra["time"][d], bin_spec, yerr=bin_err, fmt='ko', capsize=3)
                    plt.title("Spectroscopic light curve")
                    plt.xlabel("time [mjd]")
                    plt.ylabel("flux [a.u.]")
                    if save_step:
                        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_spec{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                    dpi=300, bbox_inches='tight')
                    if plot_step:
                        plt.show(block=True)
                    plt.close()

                    if (plot_ints or save_ints):
                        # Also, create Allan Variance plot.
                        fig, ax = allan_variance(spectra["time"][d], bin_spec)
                        if save_ints:
                            plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_Allan{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                        dpi=300, bbox_inches='tight')
                        if plot_ints:
                            plt.show(block=True)
                        plt.close()

        # And store.
        spec.append(spec_det)
        specerr.append(specerr_det)
        specwave.append(specwave_det)
        specbins.append(specbins_det)

    # Need these in case you don't want to bin in time.
    xpos = spectra["xpos"]
    ypos = spectra["ypos"]
    widths = spectra["widths"]
    t = spectra["time"]

    # If asked, bin in time.
    if inpt_dict["bin_time"]:
        # FIX THIS: this is going to break xpos, ypos when binning time! too bad!
        if inpt_dict["verbose"] >= 1:
            print("Binning light curves down in time...")
            
        # Get the bin size once.
        s = inpt_dict["bin_size"]

        # Start by binning time itself.
        t_new = []
        for d in range(len(t)):
            t_new.append(time_bin(t[d],s,'median'))

        t = t_new

        # We can initialize a lot of empty arrays this way.
        broadband_new = [0 for x in t]
        broaderr_new = [0 for x in t]
        xpos_new = [0 for x in t]
        ypos_new = [0 for x in t]
        widths_new = [0 for x in t]

        for d in range(len(t)):
            broadband_new[d] = time_bin(broadband[d],s,'mean')
            broaderr_new[d] = time_bin(broaderr[d],s,'quadrature')
            xpos_new[d] = time_bin(xpos[d],s,'median')
            ypos_new[d] = time_bin(ypos[d],s,'median')
            widths_new[d] = time_bin(widths[d],s,'median')

        broadband = broadband_new
        broaderr = broaderr_new
        xpos = xpos_new
        ypos = ypos_new
        widths = widths_new

        # The spec and spec_err need slightly special treatment.
        #spec_new = np.empty((t.shape[0],len(spec[0]),t.shape[1]))
        #specerr_new = np.empty((t.shape[0],len(specerr[0]),t.shape[1]))
        spec_new = []
        specerr_new = []

        for d in range(len(t)):
            specdet_new = [0 for x in range(len(spec[d]))]
            specerrdet_new = [0 for x in range(len(spec[d]))]
            for l in range(len(spec[d])):
                specdet_new[l] = time_bin(spec[d][l],s,'mean')
                specerrdet_new[l] = time_bin(specerr[d][l],s,'quadrature')
            spec_new.append(np.array(specdet_new))
            specerr_new.append(np.array(specerrdet_new))

        spec = spec_new
        specerr = specerr_new

        if (plot_step or save_step):
            # Create diagnostic plot of the binned broad-band light curve.
            for d in range(len(t)):
                plt.errorbar(t[d], broadband[d], yerr=broaderr[d], fmt='ko', capsize=3)
                plt.title("Broad-band light curve")
                plt.xlabel("time [mjd]")
                plt.ylabel("flux [a.u.]")
                if save_step:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_binnedbroadband_lc.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()

                if (plot_ints or save_ints):
                    # Also, create Allan Variance plot.
                    fig, ax = allan_variance(t[d], broadband[d])#, broaderr[d,:])
                    if save_ints:
                        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_Allanbinnedbroadband_lc.png'.format(d)),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()
                    
    # Bundle as dictionary.
    light_curves = {"broadband":broadband,
                    "broaderr":broaderr,
                    "broadwave":broadwave,
                    "broadbins":broadbins,
                    "spec":spec,
                    "specerr":specerr,
                    "specwave":specwave,
                    "specbins":specbins,
                    "time":t,
                    "xpos":xpos,
                    "ypos":ypos,
                    "widths":widths}
    
    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    return light_curves

def write_phot_to_dict(spectra, inpt_dict):
    """Converts a photometric light curve into light curve.

    Args:
        spectra (dict): its "spectrum" key has shape [detectors, time], \
        and its "time" key will be used in binning.
        inpt_dict (dict): instructions for running this step.

    Returns:
        dict: light_curves dictionary.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Converting photometric time-series to light-curve object...")

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Binning 1D spectra to produce broad-band and spectroscopic light curves...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])
    
    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Initialize some lists at the detector level.
    broadband = [] # shape detector x time
    broaderr = [] # shape detector x time
    broadwave = [] # shape detector
    broadbins = [] # shape detector
    spec = [] # shape detector x time x central_wavelength
    specerr = [] # shape detector x time x central_wavelength
    specwave = []  # shape detector x central_wavelength
    specbins = [] # shape detector x wavelength edges

    # And begin processing detectors.
    for d in tqdm(range(len(spectra["spectrum"])),#.spectrum.shape[0]),
                  desc='Processing photometric light curves from each detector...',
                  disable=(not time_ints)):
        # Load the spectra and uncertainties from this detector.
        spectrum = spectra["spectrum"][d]
        error = spectra["errors"][d]

        # The broad-band light curve is already available.
        broadband_det = spectrum
        # Broad-band uncertainties also ready.
        broaderr_det = error
        # Getting the central wavelength and bounds depends on the instrument used.
        filter_name = spectra["filters"][d]
        broadwave_det, min_wav, max_wav = translate_filter_to_waves(filter_name)
        broadbins_det = np.array([min_wav,max_wav])
        # If asked, replace outliers from the curve.
        if inpt_dict["clip_outliers"]:
            broadband_det, n_changed = running_median_filter(broadband_det,
                                                             sigma=inpt_dict["clip_outliers"],
                                                             window=inpt_dict["clip_window"])
            if inpt_dict["verbose"] == 2:
                print("Replaced {} outliers in the broadband detector light curve.".format(n_changed))
        # Store both the spectrum and each point's uncertainty.
        broadband.append(broadband_det)
        broaderr.append(broaderr_det)
        broadwave.append(broadwave_det)
        broadbins.append(broadbins_det)

        if (plot_step or save_step):
            # Create diagnostic plot of the broad-band light curve.
            plt.errorbar(spectra["time"][d], broadband_det, yerr=broaderr_det, fmt='ko', capsize=3)
            plt.title("Broad-band light curve")
            plt.xlabel("time [mjd]")
            plt.ylabel("flux [a.u.]")
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_broadband_lc.png'.format(d)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

            # Also, create Allan Variance plot.
            if (plot_ints or save_ints):
                fig, ax = allan_variance(spectra["time"][d], broadband_det)
                if save_ints:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_Allanbroadband_lc.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show(block=True)
                plt.close()
    
    # Need these in case you don't want to bin in time.
    xpos = spectra["xpos"]
    ypos = spectra["ypos"]
    widths = spectra["widths"]
    t = spectra["time"]

    # If asked, bin in time.
    if inpt_dict["bin_time"]:
        # FIX THIS: this is going to break xpos, ypos when binning time! too bad!
        if inpt_dict["verbose"] >= 1:
            print("Binning light curves down in time...")
            
        # Get the bin size once.
        s = inpt_dict["bin_size"]

        # Start by binning time itself.
        t_new = []
        for d in range(len(t)):
            t_new.append(time_bin(t[d],s,'median'))

        t = t_new

        # We can initialize a lot of empty arrays this way.
        broadband_new = [0 for x in t]
        broaderr_new = [0 for x in t]
        xpos_new = [0 for x in t]
        ypos_new = [0 for x in t]
        widths_new = [0 for x in t]

        for d in range(len(t)):
            broadband_new[d] = time_bin(broadband[d],s,'mean')
            broaderr_new[d] = time_bin(broaderr[d],s,'quadrature')
            xpos_new[d] = time_bin(xpos[d],s,'median')
            ypos_new[d] = time_bin(ypos[d],s,'median')
            widths_new[d] = time_bin(widths[d],s,'median')

        broadband = broadband_new
        broaderr = broaderr_new
        xpos = xpos_new
        ypos = ypos_new
        widths = widths_new

        if (plot_step or save_step):
            # Create diagnostic plot of the binned broad-band light curve.
            for d in range(len(t)):
                plt.errorbar(t[d], broadband[d], yerr=broaderr[d], fmt='ko', capsize=3)
                plt.title("Broad-band light curve")
                plt.xlabel("time [mjd]")
                plt.ylabel("flux [a.u.]")
                if save_step:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_binnedbroadband_lc.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()

                if (plot_ints or save_ints):
                    # Also, create Allan Variance plot.
                    fig, ax = allan_variance(t[d], broadband[d])#, broaderr[d,:])
                    if save_ints:
                        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_Allanbinnedbroadband_lc.png'.format(d)),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()
                    
    # Bundle as dictionary.
    light_curves = {"broadband":broadband,
                    "broaderr":broaderr,
                    "broadwave":broadwave,
                    "broadbins":broadbins,
                    "spec":spec,
                    "specerr":specerr,
                    "specwave":specwave,
                    "specbins":specbins,
                    "time":t,
                    "xpos":xpos,
                    "ypos":ypos,
                    "widths":widths}
    
    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    return light_curves

def time_bin(array, bin_size, mode='sum'):
    """Simple function to bin an array down in size.

    Args:
        array (np.array): the array to bin down.
        bin_size (int): how many items should go into the bin.
        mode (str, optional): how to combine the values. Options are 'sum',
        'mean', 'median', or 'quadrature'. Defaults to 'sum'.
    
    Returns:
        np.array: the input array reduced in size.
    """
    # Initialize new array as list.
    binned = []

    # Ensure no overflow.
    if bin_size > array.shape[0]:
        bin_size = array.shape[0] - 1

    # Then, iterate.
    ind = 0
    while ind+bin_size < array.shape[0]:
        trim = array[ind:ind+bin_size]
        if mode == 'sum':
            binned.append(np.ma.sum(trim))
        if mode == 'mean':
            binned.append(np.ma.mean(trim))
        if mode == 'median':
            binned.append(np.ma.median(trim))
        if mode == 'quadrature':
            binned.append(np.sqrt(np.ma.sum(np.square(trim)))/len(trim))
        ind += bin_size
    
    trim = array[ind:]
    if len(trim) == 0:
        pass
    else:
        if mode == 'sum':
            binned.append(np.ma.sum(trim))
        if mode == 'mean':
            binned.append(np.ma.mean(trim))
        if mode == 'median':
            binned.append(np.ma.median(trim))
        if mode == 'quadrature':
            binned.append(np.sqrt(np.ma.sum(np.square(trim)))/len(trim))
    # Restore array status.
    binned = np.array(binned)
    return binned

def allan_variance(time, flx):
    """Generates Allan Variance diagnostic plot.

    Args:
        time (_type_): _description_
        flx (_type_): _description_
    """
    # Get the oot residuals and std dev.
    oot_ind = int(0.25*len(flx))
    norm_flx = flx[:oot_ind]/np.median(flx[:oot_ind])
    res = est_errs(time[:oot_ind],norm_flx)
    
    # Call plot_allan util.
    fig, ax = plot_allan(res)

    return fig, ax

def est_errs(time, flx):
    try:
        result = least_squares(residuals_,
                               np.array([1,1,0,1]),
                               args=(time,flx))
    except ValueError:
        plt.scatter(time,flx)
        plt.title("somethin wrong with you")
        plt.show()
        plt.close()
        print(1/0)
    res = residuals_(result.x,time,flx)
    return res

def residuals_(fit,x,flx):
    rs = rampslope(x,fit[0],fit[1],fit[2],fit[3])
    return (flx-rs)

def rampslope(x,a,b,c,d):
    return a*np.exp(b*(x-x[0])) + c*(x-x[0]) + d

def get_rms(r):
    return (np.sum([i**2 for i in r])/len(r))**0.5

def get_GST(r, res_std, N):
    return res_std * ((len(r)/(N*(len(r)-1)))**0.5)

def running_median_filter(lc,sigma=5.0,window=None):
    """Uses a running median to filter outliers from a light curve.

    Args:
        lc (array-like): light curve to be cleaned.
        sigma (float, optional): The sigma at which to reject outliers.
        Defaults to 5.0.
        window (int, optional): The length of the window used to compute
        the running median. Defaults to None.
    
    Returns:
        array-like: cleaned light curve.
    """
    # Make window size, if needed.
    if window == None:
        window = int(0.01*len(lc))
    if window % 2 == 0:
        # Make it odd.
        window -= 1
    
    # Generate median-filtered light curve.
    smoothed_lc = ndimage.median_filter(lc,size=window,mode='nearest')

    # And replace outliers.
    corrected_lc = np.where(np.abs(lc-smoothed_lc)>sigma*np.ma.std(smoothed_lc),
                            smoothed_lc,lc)
    
    # Track number changed.
    n_changed = np.count_nonzero(np.where(corrected_lc!=lc,1,0))
    
    return lc, n_changed


def translate_filter_to_waves(filter_name):
    """Takes a str filter_name and translates it to wavelengths.

    Args:
        filter_name (str): JWST pipeline-assigned filter name.
    Returns:
        float, float, float: the central wavelength and the lower and upper limits.
    """
    # For now, MIRI imaging is supported.
    # TO DO: add NIRSpec, NIRISS, NIRCam.
    wave_data = {'F560W':(5.6,5.054,6.171,),
                 'F770W':(7.7,6.581,8.687),
                 'F1000W':(10.0,9.023,10.891),
                 'F1130W':(11.3,10.953,11.667),
                 'F1280W':(12.8,11.588,14.115),
                 'F1500W':(15.0,13.527,16.640),
                 'F1800W':(18.0,16.519,19.502),
                 'F2100W':(21.0,18.477,23.159),
                 'F2550W':(25.5,23.301,26.733),
                 }

    try:
        return wave_data[filter_name[0]]
    except KeyError:
        print('WARNING: filter_name',filter_name[0],'was not recognized as a JWST filter.')
        print('Please report this as an issue on our GitHub! We are sorry for the oversight.')
        return 