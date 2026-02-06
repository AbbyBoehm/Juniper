import os
import glob
from tqdm import tqdm

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from juniper.config.translate_config import make_planets, make_flares, make_systematics, make_ld
from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_spectra, save_s5_output
from juniper.util.plotting import plot_allan
from juniper.stage6 import plot_fit_and_res, plot_model_panel, plot_spectrum, compute_depths

def do_stage6(filepaths, outfile, outdir, steps, plot_dir):
    """Performs Stage 6 results processing on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to extract from. The files must be of type *.npy
        outfile (str): name to give to the spectra files.
        outdir (str): location of where to save the files to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 6 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 6 will operate on the following files:")
        for i, f in enumerate(filepaths):
            print(i, f)
        print("Output will be saved to {}*.txt, *.dat, and *.png formats.".format(outfile))
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_ints = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Put the plot directory into the inpt_dict and create it.
    steps["plot_dir"] = plot_dir
    if (not os.path.exists(plot_dir) and any((save_step, save_ints))):
        os.makedirs(plot_dir)

    # Decide how to handle LSQ and MCMC results.
    results = {}
    if steps["read_LSQ"]:
        # Load each lsq result.
        for f in [filepath for filepath in filepaths if "LSQ" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    if steps["read_MCMC"]:
        # Load each mcmc result.
        for f in [filepath for filepath in filepaths if "MCMC" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result
    
    # With the fits loaded, we can start making plots. We start with fits and residuals.
    if steps["plot_individual"]:
        # We want to plot each light curve in its own plot. Each key here will
        # correspond to the name of a results file.
        for key in tqdm(list(results.keys()),
                        desc='Processing fit-res for each fit...',
                        disable=(not time_ints)):
            tag = 'MCMC'
            if 'LSQ' in key:
                tag = 'LSQ'
            # Every result has keys planets, planet_errs,
            # flares, flare_errs, systematics, systematic_errs,
            # ld, ld_err, time, light_curve, errors, wavelength.
            result = results[key]

            # Each of these keys contains a "parallel ID" key which
            # encodes which spectrum it belongs to. We need to move
            # up parallel keys until we run out.

            parallel_ID = 0 # start moving up parallel IDs until we run out
            para_ID_error = False
            while not para_ID_error:
                try:
                    event_type = steps['event_type_1'][parallel_ID]
                
                    fig, ax = plot_fit_and_res.plot_fit_and_res(result['time'][parallel_ID],
                                                                result['light_curve'][parallel_ID],
                                                                result['errors'][parallel_ID],
                                                                result['planets'][str(parallel_ID+1)],
                                                                result['flares'][str(parallel_ID+1)],
                                                                result['systematics'][str(parallel_ID+1)],
                                                                result['ld'][str(parallel_ID+1)],
                                                                event_type,
                                                                steps)
                    
                    if (result['wavelength'] == 'broadband' and save_step):
                        plt.savefig(os.path.join(plot_dir,'S6_{}_broadbandID{}_fit-res{}.png'.format(outfile,parallel_ID+1,tag)),
                                    dpi=300,bbox_inches='tight')
                    elif save_ints:
                        plt.savefig(os.path.join(plot_dir,'S6_{}_{}ID{}_fit-res{}.png'.format(outfile,result['wavelength'],parallel_ID+1,tag)),
                                    dpi=300,bbox_inches='tight')
                    if (result['wavelength'] == 'broadband' and plot_step):
                        plt.show(block=True)
                    elif plot_ints:
                        plt.show(block=True)
                    plt.close()

                    parallel_ID += 1
                except IndexError:
                    para_ID_error = True

    # Reload.
    results = {}
    if steps["read_LSQ"]:
        # Load each lsq result.
        for f in [filepath for filepath in filepaths if "LSQ" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    if steps["read_MCMC"]:
        # Load each mcmc result.
        for f in [filepath for filepath in filepaths if "MCMC" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    # We can plot panels of the models, too.
    if steps["plot_components"]:
        # We want to plot each light curve's components in panels. Each key here will
        # correspond to the name of a results file.
        for key in tqdm(list(results.keys()),
                        desc='Processing model components for each fit...',
                        disable=(not time_ints)):
            tag = 'MCMC'
            if 'LSQ' in key:
                tag = 'LSQ'
            # Every result has keys planets, planet_errs,
            # flares, flare_errs, systematics, systematic_errs,
            # ld, ld_err, time, light_curve, errors, wavelength.
            result = results[key]

            # Each of these keys contains a "parallel ID" key which
            # encodes which spectrum it belongs to. We need to move
            # up parallel keys until we run out.

            parallel_ID = 0 # start moving up parallel IDs until we run out
            para_ID_error = False
            while not para_ID_error:
                try:
                    event_type = steps['event_type_1'][parallel_ID]
                    
                    fig, axes = plot_model_panel.plot_model_panel(result['time'][parallel_ID],
                                                                  result['light_curve'][parallel_ID],
                                                                  result['errors'][parallel_ID],
                                                                  result['planets'][str(parallel_ID+1)],
                                                                  result['flares'][str(parallel_ID+1)],
                                                                  result['systematics'][str(parallel_ID+1)],
                                                                  result['ld'][str(parallel_ID+1)],
                                                                  event_type,
                                                                  steps)
            
                    if (result['wavelength'] == 'broadband' and save_step):
                        plt.savefig(os.path.join(plot_dir,'S6_{}_broadbandID{}_fit-comps{}.png'.format(outfile,parallel_ID+1,tag)),
                                    dpi=300,bbox_inches='tight')
                    elif save_ints:
                        plt.savefig(os.path.join(plot_dir,'S6_{}_{}ID{}_fit-comps{}.png'.format(outfile,result['wavelength'],parallel_ID+1,tag)),
                                    dpi=300,bbox_inches='tight')
                    if (result['wavelength'] == 'broadband' and plot_step):
                        plt.show(block=True)
                    elif plot_ints:
                        plt.show(block=True)
                    plt.close()

                    parallel_ID += 1
                except IndexError:
                    para_ID_error = True

    # Reload.
    results = {}
    if steps["read_LSQ"]:
        # Load each lsq result.
        for f in [filepath for filepath in filepaths if "LSQ" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    if steps["read_MCMC"]:
        # Load each mcmc result.
        for f in [filepath for filepath in filepaths if "MCMC" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    # Plot the Allan variance of each fit.
    if steps["plot_allan_var"]:
        # We want to plot Allan variance of each fit. Each key here will
        # correspond to the name of a results file.
        for key in tqdm(list(results.keys()),
                        desc='Processing Allan variance for each fit...',
                        disable=(not time_ints)):
            tag = 'MCMC'
            if 'LSQ' in key:
                tag = 'LSQ'
            # Every result has keys planets, planet_errs,
            # flares, flare_errs, systematics, systematic_errs,
            # ld, ld_err, time, light_curve, errors, wavelength.
            result = results[key]

            # Each of these keys contains a "parallel ID" key which
            # encodes which spectrum it belongs to. We need to move
            # up parallel keys until we run out.

            parallel_ID = 0 # start moving up parallel IDs until we run out
            para_ID_error = False
            while not para_ID_error:
                try:
                    event_type = steps['event_type_1'][parallel_ID]
                
                    t_interp, lc_interp, comps, residuals = plot_fit_and_res.get_fit_and_res(result['time'][parallel_ID],
                                                                                             result['light_curve'][parallel_ID],
                                                                                             result['errors'][parallel_ID],
                                                                                             result['planets'][str(parallel_ID+1)],
                                                                                             result['flares'][str(parallel_ID+1)],
                                                                                             result['systematics'][str(parallel_ID+1)],
                                                                                             result['ld'][str(parallel_ID+1)],
                                                                                             event_type)
                    fig, ax = plot_allan(residuals)
                    
                    if (result['wavelength'] == 'broadband' and save_step):
                        plt.savefig(os.path.join(plot_dir,'S6_{}_broadbandID{}_allan-var{}.png'.format(outfile,parallel_ID+1,tag)),
                                    dpi=300,bbox_inches='tight')
                    elif save_ints:
                        plt.savefig(os.path.join(plot_dir,'S6_{}_{}ID{}_allan-var{}.png'.format(outfile,result['wavelength'],parallel_ID+1,tag)),
                                    dpi=300,bbox_inches='tight')
                    if (result['wavelength'] == 'broadband' and plot_step):
                        plt.show(block=True)
                    elif plot_ints:
                        plt.show(block=True)
                    plt.close()

                    parallel_ID += 1
                except IndexError:
                    para_ID_error = True

    # Reload.
    results = {}
    if steps["read_LSQ"]:
        # Load each lsq result.
        for f in [filepath for filepath in filepaths if "LSQ" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    if steps["read_MCMC"]:
        # Load each mcmc result.
        for f in [filepath for filepath in filepaths if "MCMC" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    # We can plot a waterfall of the fits and residuals.
    if steps["plot_waterfall"]:
        for tag in ('LSQ','MCMC'):
            result_keys = [key for key in list(results.keys()) if tag in key]

            # Each of these keys contains a "parallel ID" key which
            # encodes which spectrum it belongs to. We need to move
            # up parallel keys until we run out.

            parallel_ID = 0 # start moving up parallel IDs until we run out
            para_ID_error = False
            while not para_ID_error:
                try:
                    event_type = steps['event_type_1'][parallel_ID]
                    wavelengths, ts, lcs, lc_errs = [], [], [], []
                    t_interps, lc_interps, residualses = [], [], []
                    for key in tqdm(list(result_keys),
                                    desc='Processing fit-res for waterfall plot...',
                                    disable=(not time_ints)):
                        result = results[key]
                        if result['wavelength'] != 'broadband':
                            t_interp, lc_interp, comps, residuals = plot_fit_and_res.get_fit_and_res(result['time'][parallel_ID],
                                                                                                     result['light_curve'][parallel_ID],
                                                                                                     result['errors'][parallel_ID],
                                                                                                     result['planets'][str(parallel_ID+1)],
                                                                                                     result['flares'][str(parallel_ID+1)],
                                                                                                     result['systematics'][str(parallel_ID+1)],
                                                                                                     result['ld'][str(parallel_ID+1)],
                                                                                                     event_type)
                            wavelengths.append(float(result['wavelength']))
                            ts.append(result['time'][parallel_ID])
                            lcs.append(result['light_curve'][parallel_ID])
                            lc_errs.append(result['errors'][parallel_ID])
                            t_interps.append(t_interp)
                            lc_interps.append(lc_interp)
                            residualses.append(residuals)
                    fig, axes = plot_fit_and_res.plot_waterfall(wavelengths,
                                                                ts, lcs, lc_errs,
                                                                t_interps, lc_interps, residualses,
                                                                steps)
                    if save_step:
                        plt.savefig(os.path.join(plot_dir,'S6_{}_waterfall{}_ID{}.png'.format(outfile,tag,parallel_ID+1)),
                                    dpi=300,bbox_inches='tight')
                    if plot_step:
                        plt.show(block=True)
                    plt.close()

                    parallel_ID += 1
                except IndexError:
                    para_ID_error = True

    # Reload.
    results = {}
    if steps["read_LSQ"]:
        # Load each lsq result.
        for f in [filepath for filepath in filepaths if "LSQ" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    if steps["read_MCMC"]:
        # Load each mcmc result.
        for f in [filepath for filepath in filepaths if "MCMC" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    # Now we should compute the spectrum and save it.
    if steps["get_spectrum"]:
        for tag in ('LSQ','MCMC'):
            # Move through each parallel spectrum as available.
            parallel_ID = 0
            para_ID_error = False
            while not para_ID_error:
                try:
                    # We don't know in advance how many planets were fit,
                    # so we're going to have to keep trying until we run
                    # out of planets.
                    planet_ID = 1
                    keyError_happened = False
                    while not keyError_happened:
                        try:
                            # Get just the keys for the type of fit we are probing.
                            result_keys = [key for key in list(results.keys()) if tag in key]

                            # First check out broadband depth.
                            for key in [key for key in result_keys if results[key]['wavelength'] == 'broadband']:
                                # Get the result's planets.
                                planets = results[key]['planets'][str(parallel_ID+1)]
                                planet_errs = results[key]['planet_errs'][str(parallel_ID+1)]

                                # Get the spectrum for the current planet of interest.
                                if steps["spectrum_type"][parallel_ID] == 'rprs':
                                    depth, err = compute_depths.compute_depth_rprs(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                                if steps["spectrum_type"][parallel_ID] == 'rprs2':
                                    depth, err = compute_depths.compute_depth_rprs2(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                                if steps["spectrum_type"][parallel_ID] == 'aover':
                                    depth, err = compute_depths.compute_depth_aoverlap(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                                if steps["spectrum_type"][parallel_ID] == 'fpfs':
                                    depth, err = compute_depths.compute_depth_fpfs(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                            # And save.
                            fname = os.path.join(outdir,'S6_{}_ID{}_planet{}_broadband{}_fit{}.dat'.format(outfile,
                                                                                                           parallel_ID+1,
                                                                                                           planet_ID,
                                                                                                           steps["spectrum_type"][parallel_ID],
                                                                                                           tag))
                            with open(fname,mode='w') as f:
                                f.write("#wavelength[um] depth[{}] err[{}]\n".format(steps["spectrum_type"][parallel_ID],
                                                                                     steps["spectrum_type"][parallel_ID]))
                                f.write("{}   {}   {}\n".format('broadband',depth,err))
                                f.write('parameters\n')
                                for key in list(planets['planet{}'.format(planet_ID)].keys()):
                                    f.write("{}    {}    {}\n".format(key,
                                                                    planets['planet{}'.format(planet_ID)][key],
                                                                    planet_errs['planet{}'.format(planet_ID)][key]))

                            # Open some lists for this spectrum.
                            waves = []
                            depths = []
                            errors = []
                            for key in result_keys:
                                # Get the result's planets.
                                planets = results[key]['planets'][str(parallel_ID+1)]
                                planet_errs = results[key]['planet_errs'][str(parallel_ID+1)]
                                if results[key]['wavelength'] != 'broadband':
                                    waves.append(float(results[key]['wavelength']))

                                # Get the spectrum for the current planet of interest.
                                if steps["spectrum_type"][parallel_ID] == 'rprs':
                                    depth, err = compute_depths.compute_depth_rprs(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                                if steps["spectrum_type"][parallel_ID] == 'rprs2':
                                    depth, err = compute_depths.compute_depth_rprs2(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                                if steps["spectrum_type"] [parallel_ID]== 'aover':
                                    depth, err = compute_depths.compute_depth_aoverlap(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                                if steps["spectrum_type"][parallel_ID] == 'fpfs':
                                    depth, err = compute_depths.compute_depth_fpfs(planets['planet{}'.format(planet_ID)],
                                                                                planet_errs['planet{}'.format(planet_ID)],
                                                                                str(planet_ID))
                                
                                if results[key]['wavelength'] == 'broadband':
                                    fname = os.path.join(plot_dir,'S6_{}_ID{}_planet{}_broadband{}_fit{}.txt'.format(outfile,
                                                                                                                     parallel_ID+1,
                                                                                                                    planet_ID,
                                                                                                                    steps["spectrum_type"][parallel_ID],
                                                                                                                    tag,))
                                    with open(fname,mode='w') as f:
                                        f.write('depth err\n')
                                        f.write('{:.5f} {:.5f}'.format(depth,err))
                                else:
                                    depths.append(depth)
                                    errors.append(np.abs(err))
                            
                            # Plot, if asked.
                            if (plot_step or save_step):
                                bin_factors = steps["bin_factors"]
                                wave_bounds = steps["wave_bounds"]
                                if not steps["bin_factors"]:
                                    bin_factors = [1,]
                                if 1 not in bin_factors:
                                    bin_factors.append(1) # ensure there is always native res
                                for bin_f in bin_factors:
                                    fig, ax = plot_spectrum.plot_spectrum(waves,depths,errors,
                                                                        bin_f,wave_bounds,steps["spectrum_type"][parallel_ID])
                                    if save_step:
                                        plt.savefig(os.path.join(plot_dir,'S6_{}_ID{}_planet{}_spectrum{}_fit{}_bin{}.png'.format(outfile,
                                                                                                                                  parallel_ID+1,
                                                                                                                                  planet_ID,
                                                                                                                                  steps["spectrum_type"][parallel_ID],
                                                                                                                                  tag,
                                                                                                                                  bin_f)),
                                                    dpi=300,bbox_inches='tight')
                                    if plot_step:
                                        plt.show(block=True)
                                    plt.close()
                            
                            # And save.
                            fname = os.path.join(outdir,'S6_{}_ID{}_planet{}_spectrum{}_fit{}.dat'.format(outfile,
                                                                                                          parallel_ID+1,
                                                                                                    planet_ID,
                                                                                                    steps["spectrum_type"][parallel_ID],
                                                                                                    tag))
                            with open(fname,mode='w') as f:
                                f.write("#wavelength[um] depth[{}] err[{}]\n".format(steps["spectrum_type"][parallel_ID],
                                                                                     steps["spectrum_type"][parallel_ID]))
                                for w,d,e in zip(waves,depths,errors):
                                    f.write("{}   {}   {}\n".format(w,d,e))
                            
                            # Advance to next planet!
                            planet_ID += 1
                        except KeyError:
                            try:
                                results[key]['planets'][str(parallel_ID+1)]
                            except KeyError:
                                para_ID_error = True
                            keyError_happened = True
                    parallel_ID += 1
                except IndexError:
                    para_ID_error = True
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 6 is complete.")