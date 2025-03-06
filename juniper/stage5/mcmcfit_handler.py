import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import numpy as np
import emcee
import matplotlib.pyplot as plt

from juniper.stage5 import batman_handler, fit_handler, exotic_handler
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.cleaning import median_timeseries_filter

def mcmcfit_one(lc_time, light_curve, errors, waves, planets, flares, systematics, ld, inpt_dict, is_spec=False):
    """Performs Markov Chain Monte Carlo fitting on the given array(s) using emcee.
    Fits a single light curve. Useful for fitting spectroscopic curves.

    Args:
        lc_time (np.array): mid-exposure times for each point in the light curve.
        light_curve (np.array): median-normalized flux with time.
        errors (np.array): uncertainties associated with each data point, used
        in weighting the residuals.
        waves (np.array): used to supply the wavelength range to
        ExoTiC-LD if needed.
        planets (dict): uninitialized planet dictionaries which need to be
        initialized with the batman_handler.
        flares (dict): a series of dictionary entries describing each flaring
        event suspected to have occurred during the observation.
        systematics (dict): a series of dictionary entries describing each
        systematic model to detrend for.
        ld (dict): a dictionary describing the limb darkening model, including
        the star's physical characteristics.
        inpt_dict (dict): instructions for running this step.
        is_spec (bool, optional): whether this is a fit to a spectroscopic
        curve, in which case certain system parameters are to be locked.
        Defaults to False.
    
    Returns:
        dict, dict, dict, dict: planets, flares, systematics, and ld updated
        with fitted values.
    """
    # Copy planets, flares, systematics, and stellar limb darkening in their unmodified state.
    old_planets = planets.copy()
    old_flares = flares.copy()
    old_systematics = systematics.copy()
    old_ld = ld.copy()

    # Check if position detrending is available.
    xpos, ypos, widths = [], [], []
    if systematics["pos_detrend"]:
        xpos = systematics["xpos"]
        ypos = systematics["ypos"]

        # Smooth the positions in case the locators had trouble.
        xpos = median_timeseries_filter(xpos,sigma=3.0,kernel=21)
        ypos = median_timeseries_filter(ypos,sigma=3.0,kernel=21)
        
    if systematics["width_detrend"]:
        widths = systematics["width"]
        # Smooth the widths in case the fitter had trouble.
        widths = median_timeseries_filter(widths,sigma=3.0,kernel=21)

    # If you are doing a poly fit, set the first polynomial coefficient better.
    if systematics["poly"]:
        systematics["poly_coeffs"][0] = np.median(light_curve)

    # Check if ExoTiC-LD is being used.
    if ld["use_exotic"]:
        # We need to update our parameters then.
        ld["wavelength_range"] = np.array([np.min(waves), np.max(waves)])
        ld["ld_initialguess"] = exotic_handler.get_exotic_coefficients(ld)
    
    # (Re-)Initialize the planets, giving them the ld info they need to talk to batman properly.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, ld,
                                                     event=inpt_dict["event_type"])
    
    # Build a priors dictionary.
    params_priors = fit_handler.build_priors_dict(planets,flares,systematics,ld,
                                                  is_spec=is_spec)

    # Conveniently, the priors also tells us which keys are getting fit.
    fit_param_keys = list(params_priors.keys())

    # Translate planets, flares, systematics, and lds into a single fitting dictionary.
    params_to_fit = fit_handler.bundle_planets_flares_systematics_and_ld(planets,
                                                                         flares,
                                                                         systematics,
                                                                         ld)
    
    # Turn that into an array so emcee will accept it.
    params_array = fit_handler.dict_to_array(params_to_fit, fit_param_keys)
    
    # Set up the chains with a little bit of scatter around the initial guess.
    pos = params_array + 1e-4 * np.random.randn(inpt_dict["MCMC_chains"],
                                                params_array.shape[0])
    nwalkers, ndim = pos.shape

    # Define steps to run for.
    steps = inpt_dict["MCMC_steps"]
    if (is_spec and inpt_dict["MCMC_specsteps"]):
        steps = inpt_dict["MCMC_specsteps"]

    # Define how many steps we want to keep.
    if inpt_dict["MCMC_burnin"] >= 1:
        # If it is a whole integer, the user has asked to burn this many steps.
        discard = int(inpt_dict["MCMC_burnin"])
    else:
        # If it is a fraction, the user has asked to burn a fraction of the steps.
        discard = int(inpt_dict["MCMC_burnin"]*steps)

    # Check for parallelization.
    if inpt_dict["max_cores"] != 1:
        # Count cores that are available.
        cores = cpu_count()
        print("Found {} total cores available.".format(cores))
        if inpt_dict["max_cores"] in ('quarter','half','all'):
            # Asked for a fraction of what's available, so get that fraction.
            translate = {'quarter':0.25,'half':0.5,'all':1.0}
            n_use = int(translate[inpt_dict["max_cores"]]*cores)
        else:
            # Specified a number of cores.
            n_use = inpt_dict["max_cores"]
        if n_use > cores:
            # Don't use more cores than there are!
            n_use = cores
        print("Multiprocessing with {} cores.".format(n_use))
        pool = Pool(n_use)
    else:
        pool = None
    
    # Define the emcee sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fit_handler.log_probability,
                                    args=(params_to_fit, fit_param_keys, lc_time, light_curve, errors,
                                          params_priors, inpt_dict["priors_type"], xpos, ypos, widths),
                                    pool=pool)
    
    # And run it!
    sampler.run_mcmc(pos, steps, progress=True)#;
    
    # Pull the sampled posteriors and discard the burn-in and flatten it.
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=discard, flat=True)

    # Close up the pool, if it was made.
    if inpt_dict["max_cores"] != 1:
        pool.close()
        pool.join()

    # Get how many parameters were fit and what each one is called.
    n = np.shape(samples[:,:,0])[0]*np.shape(samples[:,:,0])[1]
    labels = [key for key in fit_param_keys]

    # Store plotting items, we may want them later.
    plotting_items = (ndim, samples, flat_samples, labels, n)

    # Turn the flattened chains into arrays.
    fitted_array, fitted_errs_array = fit_handler.get_result_from_post(ndim, flat_samples)
    
    # Turn the arrays back into dicts.
    fitted_dict = fit_handler.array_to_dict(fitted_array, params_to_fit, fit_param_keys)
    fitted_errs_dict = fit_handler.array_to_dict(fitted_errs_array, params_to_fit, fit_param_keys)

    # And then turn those back into planets, flares, and systematics.
    repack_xpos, repack_ypos, repack_widths = [],[],[]
    if "xpos" in systematics.keys():
        repack_xpos = systematics["xpos"]
    if "ypos" in systematics.keys():
        repack_ypos = systematics["ypos"]
    if "width" in systematics.keys():
        repack_widths = systematics["width"]
    planets, flares, systematics, ld = fit_handler.unpack_params_back_to_dicts(fitted_dict,
                                                                               repack_xpos,
                                                                               repack_ypos,
                                                                               repack_widths)
    
    planets_e, flares_e, systematics_e, ld_e = fit_handler.unpack_params_back_to_dicts(fitted_errs_dict,
                                                                                       repack_xpos,
                                                                                       repack_ypos,
                                                                                       repack_widths)
    
    # Fill in anything that went missing.
    planets, flares, systematics, ld = fit_handler.refill(planets,flares,systematics,ld,
                                                          old_planets,old_flares,old_systematics,old_ld)
    
    planets_e, flares_e, systematics_e, ld_e = fit_handler.refill(planets_e,flares_e,systematics_e,ld_e,
                                                                  old_planets,old_flares,old_systematics,old_ld)
    
    # Re-initialize the planets.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, ld,
                                                     event=inpt_dict["event_type"])

    # And return the fitted parameters.
    return planets, flares, systematics, ld, planets_e, flares_e, systematics_e, ld_e, plotting_items