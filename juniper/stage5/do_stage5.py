import os
import glob
from tqdm import tqdm

import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

from juniper.config.translate_config import make_planets, make_flares, make_systematics, make_ld
from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_spectra, save_s5_output
from juniper.util.plotting import plot_chains, plot_corner, plot_post, plot_nest_post
from juniper.util.cleaning import median_timeseries_filter
from juniper.stage5 import bin_light_curves, lsqfit_handler, mcmcfit_handler, nestedsampling_handler

def do_stage5(filepaths, outfile, outdir, steps, plot_dir):
    """Performs Stage 5 fitting on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to extract from. The files must be of type *_1Dspec.nc
        outfile (str): name to give to the fitted models file.
        outdir (str): location of where to save the fits to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 5 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 5 will operate on the following files:")
        for i, f in enumerate(filepaths):
            print(i, f)
        print("Output will be saved to {}_####.npy.".format(outfile))
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_ints = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Put the plot directory into the inpt_dict and create it.
    steps["plot_dir"] = plot_dir
    if (not os.path.exists(plot_dir) and any((save_step, save_ints))):
        os.makedirs(plot_dir)

    # Open all files and decide how to handle them.
    if steps["read_1D"]:
        spectra = stitch_spectra(filepaths, steps["detectors"], time_step, steps["verbose"])

        # Bin light curves.
        light_curves = bin_light_curves.bin_light_curves(spectra, steps)

        # Save light curves out.
        filename = os.path.join(outdir, 's5_{}_lightcurves.npy'.format(outfile))
        np.save(filename,light_curves)
    
    # Read the binned light curve dictionary.
    npyfile = sorted(glob.glob(os.path.join(outdir,"*lightcurves.npy")))[0]
    light_curves = np.load(npyfile,allow_pickle=True).item()

    # Begin fitting of the light curves by loading the needed dictionaries.
    planets, flares, systematics, ld = {},{},{},{}
    # We need a new dict for each spectrum that we are addressing.
    for d in range(len(light_curves["broadband"])):
        event_ID = d+1
        planets[str(event_ID)] = make_planets(steps,event_ID=event_ID)
        flares[str(event_ID)] = make_flares(steps)
        xpos = light_curves["xpos"][d]
        ypos = light_curves["ypos"][d]
        widths = light_curves["widths"][d]

        # Mandatory to normalize the position and widths data.
        #xpos -= np.median(xpos)
        #ypos -= np.median(ypos)
        #widths -= np.median(widths)

        # Mandatory to normalize the position and widths data.
        xpos = 1 + (xpos - np.min(xpos)) * (-1 - 1) / (np.max(xpos) - np.min(xpos))
        ypos = 1 + (ypos - np.min(ypos)) * (-1 - 1) / (np.max(ypos) - np.min(ypos))
        widths = 1 + (widths - np.min(widths)) * (-1 - 1) / (np.max(widths) - np.min(widths))

        # Optionally, clean and smooth the position and widths data
        # since fitter often struggles with this.
        if steps["clean_pos"]:
            xpos = median_timeseries_filter(xpos,sigma=3.0,kernel=31)
            flen = int(0.1*len(xpos))
            if flen % 2 == 0:
                flen += 1
            xpos = median_filter(xpos,flen,mode='nearest')

            ypos = median_timeseries_filter(ypos,sigma=3.0,kernel=31)
            flen = int(0.1*len(ypos))
            if flen % 2 == 0:
                flen += 1
            ypos = median_filter(ypos,flen,mode='nearest')
        if steps["clean_widths"]:
            widths = median_timeseries_filter(widths,sigma=3.0,kernel=31)
            flen = int(0.1*len(widths))
            if flen % 2 == 0:
                flen += 1
            widths = median_filter(widths,flen,mode='nearest')
        
        if (plot_step or save_step):
            # Plot the detrending variables for reference.
            for var, var_name in zip((xpos,ypos,widths),("dispersion","cross-dispersion","widths")):
                plt.figure(figsize=(10,5))
                plt.scatter(light_curves["time"][d],var,color='k')
                plt.xlabel('Exposure Time [BJD TDB]')
                plt.ylabel(f'Trend: {var_name}')
                plt.tick_params(which='both',axis='both',direction='in')
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+f"_ID{d}_trend-{var_name}.png"),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()
        
        systematics[str(event_ID)] = make_systematics(steps,
                                                      xpos=xpos,
                                                      ypos=ypos,
                                                      widths=widths,
                                                      event_ID=str(event_ID))

        # This is the only one that doesn't need event_ID supplied,
        # because the stellar parameters and model choice are fixed.
        ld[str(event_ID)] = make_ld(steps)

    # Now fit the broadband curves.
    if steps["fit_broadband"]:
        # Fit whatever broad-band light curves are supplied.
        # If more than one is supplied, we will fit in parallel.
        if steps["use_LSQ"]:
            if steps["verbose"] == 2:
                print("Linear least squares fitting to all supplied broadband data...")
            planets, flares, systematics, ld = lsqfit_handler.lsqfit(exp_times=light_curves["time"],
                                                                     light_curve=light_curves["broadband"],
                                                                     errors=light_curves["broaderr"],
                                                                     wavelengths=light_curves["broadbins"],
                                                                     planets=planets, flares=flares,
                                                                     systematics=systematics, ld=ld,
                                                                     inpt_dict=steps, is_spec=False,
                                                                     show_guess_plot=plot_step,
                                                                     save_guess_plot=save_step,
                                                                     show_estimate=plot_ints,
                                                                     save_estimate=save_ints,
                                                                     plot_dir=plot_dir,outfile=outfile,wavestr=None)
            
            # Save output. Needs to be formatted as if there is more than one dimension.
            planets_err, flares_err, systematics_err, ld_err = planets, flares, systematics, ld
            save_s5_output(planets, planets_err, flares, flares_err,
                           systematics, systematics_err, ld, ld_err,
                           light_curves["time"], light_curves["broadband"],light_curves["broaderr"],
                           'broadband', outfile+"_broadbandLSQ", outdir)
        
        # Now refine those linear fits with MCMC, or just go straight to MCMC if desired.
        if steps["use_MCMC"]:
            if steps["verbose"] == 2:
                print("Markov Chain Monte Carlo fitting to all supplied broadband data...")
            planets, flares, systematics, ld, \
            planets_err, flares_err, systematics_err, ld_err, \
            plotting_items = mcmcfit_handler.mcmcfit(exp_times=light_curves["time"],
                                                     light_curve=light_curves["broadband"],
                                                     errors=light_curves["broaderr"],
                                                     wavelengths=light_curves["broadbins"],
                                                     planets=planets, flares=flares,
                                                     systematics=systematics, ld=ld,
                                                     inpt_dict=steps, is_spec=False,
                                                     show_guess_plot=plot_step,
                                                     save_guess_plot=save_step,
                                                     show_estimate=plot_ints,
                                                     save_estimate=save_ints,
                                                     plot_dir=plot_dir,outfile=outfile,wavestr=None)
            
            # Save output.
            save_s5_output(planets, planets_err, flares, flares_err,
                           systematics, systematics_err, ld, ld_err,
                           light_curves["time"], light_curves["broadband"],light_curves["broaderr"],
                           'broadband', outfile+"_broadbandMCMC", outdir)
            
            # Plot, if asked.
            if (plot_step or save_step):
                # Unpack plotting items.
                ndim, samples, flat_samples, labels, n = plotting_items

                # Plot corners.
                fig = plot_corner(flat_samples,labels)
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandMCMC-corner.png"),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()

                if (plot_ints or save_ints):
                    # Plot posteriors.
                    fig, ax = plot_post(ndim,samples,labels,n)
                    if save_ints:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandMCMC-posterior.png"),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()

                    # Plot chains.
                    fig, ax = plot_chains(ndim,samples,labels)
                    if save_ints:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandMCMC-chains.png"),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()
        
        # Alternatively or additionally, use nested sampling to fit the broadband data.
        if steps["use_nested"]:
            if steps["verbose"] == 2:
                print("Dynesty nested sampling fitting to all supplied broadband data...")
            planets, flares, systematics, ld, \
            planets_err, flares_err, systematics_err, ld_err, \
            plotting_items = nestedsampling_handler.nestfit(exp_times=light_curves["time"],
                                                            light_curve=light_curves["broadband"],
                                                            errors=light_curves["broaderr"],
                                                            wavelengths=light_curves["broadbins"],
                                                            planets=planets, flares=flares,
                                                            systematics=systematics, ld=ld,
                                                            inpt_dict=steps, is_spec=False,
                                                            show_guess_plot=plot_step,
                                                            save_guess_plot=save_step,
                                                            show_estimate=plot_ints,
                                                            save_estimate=save_ints,
                                                            plot_dir=plot_dir,outfile=outfile,wavestr=None)
            
            # Save output.
            save_s5_output(planets, planets_err, flares, flares_err,
                           systematics, systematics_err, ld, ld_err,
                           light_curves["time"], light_curves["broadband"],light_curves["broaderr"],
                           'broadband', outfile+"_broadbandnested", outdir)
            
            # Plot, if asked.
            if (plot_step or save_step):
                # Unpack plotting items.
                ndim, samples, labels = plotting_items

                # Plot corners.
                fig = plot_corner(samples,labels)
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandnested-corner.png"),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()

                if (plot_ints or save_ints):
                    # Plot posteriors.
                    fig, ax = plot_nest_post(ndim,samples,labels)
                    if save_ints:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandnested-posterior.png"),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()
            
    # Then fit the spectroscopic curves.
    if steps["fit_spec"]:
        # Load planets, flares, systematics, and ld from a fitted model, if available.
        if not steps["fit_broadband"]:
            try:
                result = np.load(os.path.join(outdir,outfile+"_broadbandMCMC.npy"), allow_pickle=True).item()
                planets, flares, systematics, ld = (result['planets'],result['flares'],
                                                    result['systematics'],result['ld'])
                planets_err, flares_err, systematics_err, ld_err = (result['planet_errs'],result['flare_errs'],
                                                                    result['systematic_errs'],result['ld_err'])
                print("MCMC broadband fit successfully loaded.")
            except FileNotFoundError:
                print("No MCMC results found, proceeding with initial guesses.")
        # Reserve planets, flares, systematics, and ld originals.
        planets0 = planets.copy()
        flares0 = flares.copy()
        systematics0 = systematics.copy()
        ld0 = ld.copy()
        planets_err0 = planets_err.copy()
        flares_err0 = flares_err.copy()
        systematics_err0 = systematics_err.copy()
        ld_err0 = ld_err.copy()

        # Treatments of multiple possible visits/detectors is complex. Let's break it down.
        # First, we have to check if all the wavelengths are equal.
        detector_wavelengths = []
        for i in range(len(light_curves["specbins"])):
            detector_wavelengths.append(light_curves["specbins"][i])
        detectors_are_same_bins = False
        if all(np.all(np.abs([a1-a2 for a1, a2 in zip(detector_wavelengths[0],dlist)])<0.1) for dlist in detector_wavelengths):
            detectors_are_same_bins = True
        if steps["verbose"] == 2:
            if detectors_are_same_bins:
                print("Detectors were found to have matching wavelength bins to within 0.1 micron; treatment will be parallel.")
            else:
                print("Detector wavelengths do not match to within 0.1 micron; treatment will be serial.")

        if detectors_are_same_bins:
            # Then we can treat this stuff in parallel fit.
            # Of course, how that manifests will depend on your preserve calls.

            # Iterate over each wavelength bin.
            for i in range(len(light_curves["spec"][0])):
                # Get the parallelised spectra dict for this wavelength band.
                light_curve, errors, wavelengths = [], [], []
                try:
                    for j in range(len(light_curves["spec"])):
                        light_curve.append(light_curves["spec"][j][i])
                        errors.append(light_curves["specerr"][j][i])
                        wavelengths.append(light_curves["specbins"][j][i])
                except IndexError:
                    if steps["verbose"] >= 1:
                        print("Wavelength",light_curves["specbins"][0][i],"does not exist in detector",j+1,".")
                    continue
                
                wavestr = '{:.3f}'.format(np.mean(wavelengths[0]))

                if steps["use_LSQ"]:
                    if steps["verbose"] == 2:
                        print("Linear least squares fitting to supplied spectroscopic light curve {} micron in parallel...".format(wavestr))
                    # Now we lsqfit this.
                    planets, flares, systematics, ld = lsqfit_handler.lsqfit(exp_times=light_curves["time"],
                                                                             light_curve=light_curve,
                                                                             errors=errors,
                                                                             wavelengths=wavelengths,
                                                                             planets=planets, flares=flares,
                                                                             systematics=systematics, ld=ld,
                                                                             inpt_dict=steps, is_spec=True,
                                                                             show_guess_plot=plot_step,
                                                                             save_guess_plot=save_step,
                                                                             show_estimate=plot_ints,
                                                                             save_estimate=save_ints,
                                                                             plot_dir=plot_dir,outfile=outfile,wavestr=wavestr)
                
                    # Save output. Needs to be formatted as if there is more than one dimension.
                    planets_err, flares_err, systematics_err, ld_err = planets, flares, systematics, ld
                    save_s5_output(planets, planets_err, flares, flares_err,
                                   systematics, systematics_err, ld, ld_err,
                                   light_curves["time"], light_curve, errors,
                                   wavestr, outfile+"_spec{}LSQ".format(wavestr), outdir)
                if steps["use_MCMC"]:
                    if steps["verbose"] == 2:
                        print("Markov Chain Monte Carlo fitting to supplied spectroscopic light curve {} micron in parallel...".format(wavestr))
                    # Now we mcmcfit this.
                    planets, flares, systematics, ld, \
                    planets_err, flares_err, systematics_err, ld_err, \
                    plotting_items = mcmcfit_handler.mcmcfit(exp_times=light_curves["time"],
                                                            light_curve=light_curve,
                                                            errors=errors,
                                                            wavelengths=wavelengths,
                                                            planets=planets, flares=flares,
                                                            systematics=systematics, ld=ld,
                                                            inpt_dict=steps, is_spec=True,
                                                            show_guess_plot=plot_step,
                                                            save_guess_plot=save_step,
                                                            show_estimate=plot_ints,
                                                            save_estimate=save_ints,
                                                            plot_dir=plot_dir,outfile=outfile,wavestr=wavestr)
                    
                    # Save output.
                    save_s5_output(planets, planets_err, flares, flares_err,
                                   systematics, systematics_err, ld, ld_err,
                                   light_curves["time"], light_curve, errors,
                                   wavestr, outfile+"_spec{}MCMC".format(wavestr), outdir)
                    
                    # Plot, if asked.
                    if (plot_step or save_step):
                        # Unpack plotting items.
                        ndim, samples, flat_samples, labels, n = plotting_items
                        
                        # Plot corners.
                        fig = plot_corner(flat_samples,labels)
                        if save_step:
                            plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-corner.png".format(wavestr)),
                                        dpi=300, bbox_inches='tight')
                        if plot_step:
                            plt.show(block=True)
                        plt.close()

                        if (plot_ints or save_ints):
                            # Plot posteriors.
                            fig, ax = plot_post(ndim,samples,labels,n)
                            if save_ints:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-posterior.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_ints:
                                plt.show(block=True)
                            plt.close()

                            # Plot chains.
                            fig, ax = plot_chains(ndim,samples,labels)
                            if save_ints:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-chains.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_ints:
                                plt.show(block=True)
                            plt.close()

                # Alternatively or additionally, fit with dynesty.
                if steps["use_nested"]:
                    if steps["verbose"] == 2:
                        print("Dynesty nested sampling fitting to supplied spectroscopic light curve {} micron in parallel...".format(wavestr))
                    planets, flares, systematics, ld, \
                    planets_err, flares_err, systematics_err, ld_err, \
                    plotting_items = nestedsampling_handler.nestfit(exp_times=light_curves["time"],
                                                                    light_curve=light_curve,
                                                                    errors=errors,
                                                                    wavelengths=wavelengths,
                                                                    planets=planets, flares=flares,
                                                                    systematics=systematics, ld=ld,
                                                                    inpt_dict=steps, is_spec=True,
                                                                    show_guess_plot=plot_step,
                                                                    save_guess_plot=save_step,
                                                                    show_estimate=plot_ints,
                                                                    save_estimate=save_ints,
                                                                    plot_dir=plot_dir,outfile=outfile,wavestr=wavestr)
            
                    # Save output
                    save_s5_output(planets, planets_err, flares, flares_err,
                                   systematics, systematics_err, ld, ld_err,
                                   light_curves["time"], light_curve, errors,
                                   wavestr, outfile+"_spec{}nested".format(wavestr), outdir)
            
                    # Plot, if asked.
                    if (plot_step or save_step):
                        # Unpack plotting items.
                        ndim, samples, labels = plotting_items

                        # Plot corners.
                        fig = plot_corner(samples,labels)
                        if save_step:
                            plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}nested-corner.png".format(wavestr)),
                                        dpi=300, bbox_inches='tight')
                        if plot_step:
                            plt.show(block=True)
                        plt.close()
                        
                        if (plot_ints or save_ints):
                            # Plot posteriors.
                            fig, ax = plot_nest_post(ndim,samples,labels)
                            if save_ints:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}nested-posterior.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_ints:
                                plt.show(block=True)
                            plt.close()
                
                # Reset planets, etc. to originals.
                planets, flares, systematics, ld = planets0, flares0, systematics0, ld0
                planets_err, flares_err, systematics_err, ld_err = planets_err0, flares_err0, systematics_err0, ld_err0
                    
        else:
            # We need to treat one light curve at a time, one detector at a time.
            # Again, how that manifests will depend on your preserve calls.

            # Iterate over each detector.
            for d in range(len(light_curves["spec"])):
                # Iterate over each wavelength bin.
                for i in range(len(light_curves["spec"][d])):
                    # Get the detector's spectrum dict for this wavelength band.
                    light_curve = [light_curves["spec"][d][i],]
                    errors = [light_curves["specerr"][d][i],]
                    wavelengths = [light_curves["specbins"][d][i],]
                    
                    wavestr = '{:.3f}'.format(np.mean(wavelengths[0]))

                    # Get the specific planets, flares, systematics, ld we need here,
                    # and ignore the other entries. We'll call this one "1".
                    planets = {"1":planets[str(d+1)]}
                    flares = {"1":flares[str(d+1)]}
                    systematics = {"1":systematics[str(d+1)]}
                    ld = {"1":ld[str(d+1)]}

                    if steps["use_LSQ"]:
                        if steps["verbose"] == 2:
                            print("Linear least squares fitting to supplied spectroscopic light curve {} micron in series...".format(wavestr))
                        # Now we lsqfit this.
                        planets, flares, systematics, ld = lsqfit_handler.lsqfit(exp_times=[light_curves["time"][d],],
                                                                                 light_curve=light_curve,
                                                                                 errors=errors,
                                                                                 wavelengths=wavelengths,
                                                                                 planets=planets, flares=flares,
                                                                                 systematics=systematics, ld=ld,
                                                                                 inpt_dict=steps, is_spec=True,
                                                                                 show_guess_plot=plot_step,
                                                                                 save_guess_plot=save_step,
                                                                                 show_estimate=plot_ints,
                                                                                 save_estimate=save_ints,
                                                                                 plot_dir=plot_dir,outfile=outfile,wavestr=wavestr)
                    
                        # Save output. Needs to be formatted as if there is more than one dimension.
                        planets_err, flares_err, systematics_err, ld_err = planets, flares, systematics, ld
                        save_s5_output(planets, planets_err, flares, flares_err,
                                    systematics, systematics_err, ld, ld_err,
                                    light_curves["time"], light_curve, errors,
                                    wavestr, outfile+"_spec{}LSQ_ID{}".format(wavestr,d+1), outdir)
                    if steps["use_MCMC"]:
                        if steps["verbose"] == 2:
                            print("Markov Chain Monte Carlo fitting to supplied spectroscopic light curve {} micron in series...".format(wavestr))
                        # Now we mcmcfit this.
                        planets, flares, systematics, ld, \
                        planets_err, flares_err, systematics_err, ld_err, \
                        plotting_items = mcmcfit_handler.mcmcfit(exp_times=[light_curves["time"][d],],
                                                                 light_curve=light_curve,
                                                                 errors=errors,
                                                                 wavelengths=wavelengths,
                                                                 planets=planets, flares=flares,
                                                                 systematics=systematics, ld=ld,
                                                                 inpt_dict=steps, is_spec=True,
                                                                 show_guess_plot=plot_step,
                                                                 save_guess_plot=save_step,
                                                                 show_estimate=plot_ints,
                                                                 save_estimate=save_ints,
                                                                 plot_dir=plot_dir,outfile=outfile,wavestr=wavestr)
                        
                        # Save output.
                        save_s5_output(planets, planets_err, flares, flares_err,
                                       systematics, systematics_err, ld, ld_err,
                                       light_curves["time"], light_curve, errors,
                                       wavestr, outfile+"_spec{}MCMC_ID{}".format(wavestr,d+1), outdir)
                        
                        # Plot, if asked.
                        if (plot_step or save_step):
                            # Unpack plotting items.
                            ndim, samples, flat_samples, labels, n = plotting_items

                            # Plot corners.
                            fig = plot_corner(flat_samples,labels)
                            if save_step:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}_ID{}MCMC-corner.png".format(wavestr,d+1)),
                                            dpi=300, bbox_inches='tight')
                            if plot_step:
                                plt.show(block=True)
                            plt.close()

                            if (plot_ints or save_ints):
                                # Plot posteriors.
                                fig, ax = plot_post(ndim,samples,labels,n)
                                if save_ints:
                                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC_ID{}-posterior.png".format(wavestr,d+1)),
                                                dpi=300, bbox_inches='tight')
                                if plot_ints:
                                    plt.show(block=True)
                                plt.close()

                                # Plot chains.
                                fig, ax = plot_chains(ndim,samples,labels)
                                if save_ints:
                                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}_ID{}MCMC-chains.png".format(wavestr,d+1)),
                                                dpi=300, bbox_inches='tight')
                                if plot_ints:
                                    plt.show(block=True)
                                plt.close()
                    
                    # Alternatively or additionally, fit with dynesty.
                    if steps["use_nested"]:
                        if steps["verbose"] == 2:
                            print("Dynesty nested sampling fitting to supplied spectroscopic light curve {} micron in series...".format(wavestr))
                        planets, flares, systematics, ld, \
                        planets_err, flares_err, systematics_err, ld_err, \
                        plotting_items = nestedsampling_handler.nestfit(exp_times=light_curves["time"],
                                                                        light_curve=light_curve,
                                                                        errors=errors,
                                                                        wavelengths=wavelengths,
                                                                        planets=planets, flares=flares,
                                                                        systematics=systematics, ld=ld,
                                                                        inpt_dict=steps, is_spec=True,
                                                                        show_guess_plot=plot_step,
                                                                        save_guess_plot=save_step,
                                                                        show_estimate=plot_ints,
                                                                        save_estimate=save_ints,
                                                                        plot_dir=plot_dir,outfile=outfile,wavestr=wavestr)
                
                        # Save output
                        save_s5_output(planets, planets_err, flares, flares_err,
                                    systematics, systematics_err, ld, ld_err,
                                    light_curves["time"], light_curve, errors,
                                    wavestr, outfile+"_spec{}nested_ID{}".format(wavestr,d+1), outdir)
                
                        # Plot, if asked.
                        if (plot_step or save_step):
                            # Unpack plotting items.
                            ndim, samples, labels = plotting_items
                            
                            # Plot corners.
                            fig = plot_corner(samples,labels)
                            if save_step:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}_ID{}nested-corner.png".format(wavestr,d+1)),
                                            dpi=300, bbox_inches='tight')
                            if plot_step:
                                plt.show(block=True)
                            plt.close()

                            if (plot_ints or save_ints):
                                # Plot posteriors.
                                fig, ax = plot_nest_post(ndim,samples,labels)
                                if save_ints:
                                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}_ID{}nested-corner.png".format(wavestr,d+1)),
                                                dpi=300, bbox_inches='tight')
                                if plot_ints:
                                    plt.show(block=True)
                                plt.close()
                        
                    # Reset planets, etc. to originals.
                    planets, flares, systematics, ld = planets0, flares0, systematics0, ld0
                    planets_err, flares_err, systematics_err, ld_err = planets_err0, flares_err0, systematics_err0, ld_err0

        '''
        for detector in tqdm(len(light_curves["broadband"]),
                             desc="Processing each detector's spectrum...",
                             disable=(not time_step)):
            for wavelength in tqdm(range(light_curves["specwave"][detector].shape[0]),
                                   desc='Processing each wavelength in that detector...',
                                   disable=(not time_ints)):
                # Fetch the relevant curves and values.
                time = light_curves["time"][detector]
                light_curve = light_curves["spec"][detector][wavelength]
                errors = light_curves["specerr"][detector][wavelength]
                waves = light_curves["specbins"][detector][wavelength]
                wavestr = np.round(light_curves["specwave"][detector][wavelength],3)

                # Confirm it's not empty.
                if all([i == 0 for i in light_curve]) or all([np.isnan(i) for i in light_curve]):
                    print("Light curve number {} is an empty light curve, passing...".format(wavelength))
                    continue

                # Confirm it spans more than one wavelength.
                if waves[0] == waves[1]:
                    print("Light curve number {} does not span a wavelength range, passing...".format(wavelength))
                    continue

                # Confirm it is not spanning wavelength 0 nm, for which no photons exist.
                if 0 in waves:
                    print("Light curve number {} spans a wavelength range where photons don't exist, passing...".format(wavelength))
                    continue

                # First, LSQ.
                try:
                    if steps["use_LSQ"]:
                        if steps["verbose"] == 2:
                            print("Linear least squares fitting to single spectroscopic light curve..")
                        planets, flares, systematics, ld = lsqfit_handler.lsqfit_one(lc_time=time,
                                                                            light_curve=light_curve,
                                                                            errors=errors,
                                                                            waves=waves,
                                                                            planets=planets,flares=flares,
                                                                            systematics=systematics,ld=ld,
                                                                            inpt_dict=steps,
                                                                            is_spec=True)
                        # Save output.
                        planets_err, flares_err, systematics_err, ld_err = planets, flares, systematics, ld
                        save_s5_output(planets, planets_err, flares, flares_err,
                                    systematics, systematics_err, ld, ld_err,
                                    time, light_curve, errors, wavestr,
                                    outfile+"_spec{}LSQ".format(wavestr), outdir)
                except:
                    if steps["verbose"] == 2:
                        print("Linear least squares fitting failed, likely due to batman convergence failure.")
                # Then MCMC.
                try:
                    if steps["use_MCMC"]:
                        if steps["verbose"] == 2:
                            print("Markov Chain Monte Carlo fitting to single spectroscopic light curve...")
                        planets, flares, systematics, ld, p_err, f_err, s_err, l_err, plotting_items = mcmcfit_handler.mcmcfit_one(lc_time=time,
                                                                                                                           light_curve=light_curve,
                                                                                                                           errors=errors,
                                                                                                                           waves=waves,
                                                                                                                           planets=planets,flares=flares,
                                                                                                                           systematics=systematics,ld=ld,
                                                                                                                           inpt_dict=steps,
                                                                                                                           is_spec=True)
                        
                        # Fill in errors for things that were fixed here.
                        for planet in list(planets.keys()):
                            for key in list(planets[planet].keys()):
                                if planets[planet][key] == p_err[planet][key]:
                                    # Get the broadband error to replace it with.
                                    p_err[planet][key] = p_err0[planet][key]
                        for flare in list(flares.keys()):
                            for key in list(flares[flare].keys()):
                                if flares[flare][key] == f_err[flare][key]:
                                    # Get the broadband error to replace it with.
                                    f_err[flare][key] = f_err0[flare][key]
                
                        # Save output.
                        save_s5_output(planets, p_err, flares, f_err,
                                        systematics, s_err, ld, l_err,
                                        time, light_curve, errors, wavestr,
                                        outfile+"_spec{}MCMC".format(wavestr), outdir)
                        
                        # Plot, if asked.
                        if (plot_ints or save_ints):
                            # Unpack plotting items.
                            ndim, samples, flat_samples, labels, n = plotting_items
                            labels = [str.replace(a, "_prior", "") for a in labels] # clip off the prior tag
                            # Plot posteriors.
                            fig, ax = plot_post(ndim,samples,labels,n)
                            if save_ints:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-posterior.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_ints:
                                plt.show(block=True)
                            plt.close()

                            # Plot corners.
                            fig = plot_corner(flat_samples,labels)
                            if save_ints:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-corner.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_ints:
                                plt.show(block=True)
                            plt.close()

                            # Plot chains.
                            fig, ax = plot_chains(ndim,samples,labels)
                            if save_step:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-chains.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_step:
                                plt.show(block=True)
                            plt.close()


                except:
                    if steps["verbose"] == 2:
                        print("Markov Chain Monte Carlo fititng failed, likely due to batman convergence failure.")
                # Reset planets, etc. to originals.
                planets, flares, systematics, ld = planets0, flares0, systematics0, ld0
        '''


    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 5 is complete.")