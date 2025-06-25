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

def mcmcfit(exp_times, light_curve, errors, wavelengths,
            planets, flares, systematics, ld,
            inpt_dict, is_spec=False):
    """Performs Markov Chain Monte Carlo fitting on the given array(s) using scipy.
    
    Args:
        exp_times (np.array): mid-exposure times for each point in the light curve.
        light_curve (np.array): 2D total flux in each integration in each curve.
        errors (np.array): uncertainties associated with each data point, used
        in weighting the residuals.
        wavelengths (np.array): used to supply the wavelength range to
        ExoTiC-ld if needed.
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
    # Copy original dicts to preserve prior info.
    originals = {}
    for key in list(planets.keys()):
        originals[key] = {"planets":planets[key].copy(),
                          "flares":flares[key].copy(),
                          "systematics":systematics[key].copy(),
                          "ld":ld[key].copy()}
    
    # If you are doing a poly fit, set the first polynomial coefficient better.
    for i,key in enumerate(list(systematics.keys())):
        if systematics[key]["poly"]:
            systematics[key]["poly_coeffs"][0] = np.median(light_curve[i])

    # Check if ExoTiC-LD is being used.
    for i,key in enumerate(list(ld.keys())):
        # A key we only need if using ExoTiC-LD, but the bundler expects it to be present.
        ld[key]["wavelength_range"] = np.array([np.min(wavelengths[i]),
                                                np.max(wavelengths[i])])
        if ld[key]["use_exotic"]:
            ld[key]["ld_initialguess"] = exotic_handler.get_exotic_coefficients(ld[key])
            ld[key]["ld_coeffs"] = ld[key]["ld_initialguess"]

    # (Re-)Initialize the planets, giving them the ld info they need to talk to batman properly.
    for i,key in enumerate(list(planets.keys())):
        planets[key] = batman_handler.batman_init_all_planets(exp_times[i], planets[key], ld[key],
                                                              event=inpt_dict["event_type_"+key])

    # Build a priors dictionary, and log information about what is getting fit.
    params_priors, fit_or_not = fit_handler.build_priors_dict(planets,flares,systematics,ld,
                                                              is_spec=is_spec)
    
    # Unpack priors into a list of lists.
    unpack_priors = []
    for superdict_key in list(planets.keys()):
        for key in list(params_priors[superdict_key].keys()):
            unpack_priors.append(params_priors[superdict_key][key])

    # Translate planets, flares, systematics, and lds into a single fitting dictionary.
    bundled_params = {}
    for i,key in enumerate(list(planets.keys())):
        # One entry per spec being fitted.
        bundled_params[key] = fit_handler.bundle_planets_flares_systematics_and_ld(planets[key],
                                                                                   flares[key],
                                                                                   systematics[key],
                                                                                   ld[key])
        
    # Turn that into an array so scipy will accept it.
    params_array = fit_handler.dict_to_array(bundled_params, fit_or_not)

    # Parse the preserve args.
    preserve_timing = inpt_dict["preserve_timing"]
    preserve_depth = inpt_dict["preserve_depth"]
    preserve_orbit = inpt_dict["preserve_orbit"]

    # Set up the chains with a little bit of scatter around the initial guess.
    if int(inpt_dict["MCMC_chains"]) < 2*params_array.shape[0]:
        new_n_walkers = 2*params_array.shape[0]
        if inpt_dict["verbose"] >= 1:
            print("Input n. chains {} was less than twice the number of parameters, raising value to {}.".format(inpt_dict["MCMC_chains"],new_n_walkers))
        inpt_dict["MCMC_chains"] = new_n_walkers
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
        if inpt_dict["verbose"] >= 1:
            print("Found {} total cores available.".format(cores))
        if inpt_dict["max_cores"] in ('quarter','half','all'):
            # Asked for a fraction of what's available, so get that fraction.
            translate = {'quarter':0.25,'half':0.5,'all':1.0}
            n_use = int(translate[inpt_dict["max_cores"]]*cores)
        else:
            # Specified a number of cores.
            n_use = int(inpt_dict["max_cores"])
        if n_use > cores:
            # Don't use more cores than there are!
            n_use = cores
        if inpt_dict["verbose"] >= 1:
            print("Multiprocessing with {} out of {} cores.".format(n_use,cores))
        pool = Pool(n_use)
    else:
        pool = None
    
    # Define the emcee sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fit_handler.log_probability,
                                    args=(bundled_params, fit_or_not, exp_times, light_curve, errors,
                                          unpack_priors, inpt_dict["priors_type"],
                                          preserve_timing, preserve_depth, preserve_orbit),
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

    # Turn the flattened chains into arrays.
    fitted_array, fitted_errs_array = fit_handler.get_result_from_post(ndim, flat_samples)

    # Turn both back into dicts.
    fitted_dict = fit_handler.array_to_dict(fitted_array, bundled_params, fit_or_not)
    fitted_errs_dict = fit_handler.array_to_dict(fitted_errs_array, bundled_params, fit_or_not)

    # Repack the dict back into planets, flares, systematics, and lds.
    planets, flares, systematics, ld = {}, {}, {}, {}
    for key in list(fitted_dict.keys()):
        planets[key], flares[key], systematics[key], ld[key] = fit_handler.unpack_params_back_to_dicts(fitted_dict[key],
                                                                                                       originals[key])
        
    # Repeat preserve calls.
    planets = fit_handler.preservation(planets,
                                       preserve_timing, preserve_depth, preserve_orbit)

    # Same for errors.
    planets_err, flares_err, systematics_err, ld_err = {}, {}, {}, {}
    for key in list(fitted_errs_dict.keys()):
        planets_err[key], flares_err[key], systematics_err[key], ld_err[key] = fit_handler.unpack_params_back_to_dicts(fitted_errs_dict[key],
                                                                                                                       originals[key])
    
    # Repeat preserve calls.
    planets_err = fit_handler.preservation(planets_err,
                                           preserve_timing, preserve_depth, preserve_orbit)

    # Re-initialize the planets to get the updated models into place.
    for i,key in enumerate(list(planets.keys())):
        planets[key] = batman_handler.batman_init_all_planets(exp_times[i], planets[key], ld[key],
                                                              event=inpt_dict["event_type_"+key])

    # Get what each one is called.
    labels = []
    for superdict_key in list(fit_or_not.keys()):
        for key in list(fit_or_not[superdict_key].keys()):
            if fit_or_not[superdict_key][key]:
                labels.append(str.replace(key,'_prior',''))
    labels = np.array(labels,dtype=str)

    # Delete repeat parameters as needed.
    delete_indices = []
    if preserve_timing:
        # Every spare copy of t_primN, t_secoN must be deleted.
        for delete_parameter in ("t_prim","t_seco"):
            for planet_N in range(1,100):
                parameter_indices = [i for i, x in enumerate(labels) if '{}{}'.format(delete_parameter,planet_N) in x]
                for i in range(1,len(parameter_indices)): # the 1 lets us skip the first instance
                    delete_indices.append(parameter_indices[i])
    if preserve_depth:
        # Every spare copy of rpN, fpN must be deleted.
        for delete_parameter in ("rp","fp"):
            for planet_N in range(1,100):
                parameter_indices = [i for i, x in enumerate(labels) if '{}{}'.format(delete_parameter,planet_N) in x]
                for i in range(1,len(parameter_indices)): # the 1 lets us skip the first instance
                    delete_indices.append(parameter_indices[i])
    if preserve_orbit:
        # Every spare copy of aorN, periodN, inclN, eccN, longitudeN must be deleted.
        for delete_parameter in ("aor","period","incl","ecc","longitude"):
            for planet_N in range(1,100):
                parameter_indices = [i for i, x in enumerate(labels) if '{}{}'.format(delete_parameter,planet_N) in x]
                for i in range(1,len(parameter_indices)): # the 1 lets us skip the first instance
                    delete_indices.append(parameter_indices[i])
    # Now delete the indices, if any were found.
    if delete_indices:
        if inpt_dict["verbose"] == 2:
            print("Found {} repeat indices to delete as part of preserve arguments.".format(len(delete_indices)))
        delete_indices = np.array([int(i) for i in delete_indices])
        samples = np.delete(samples,obj=delete_indices,axis=2)
        flat_samples = np.delete(flat_samples,obj=delete_indices,axis=1)
        labels = np.delete(labels,obj=delete_indices,axis=0)

    # Get how many parameters were fit.
    n = np.shape(samples[:,:,0])[0]*np.shape(samples[:,:,0])[1]

    # Update ndim, necessary if there were deletions.
    ndim = samples.shape[2]

    # Store plotting items, we may want them later.
    plotting_items = (ndim, samples, flat_samples, labels, n)
    
    # And return the fitted parameters.
    return planets, flares, systematics, ld, planets_err, flares_err, systematics_err, ld_err, plotting_items

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
    if systematics["disp_detrend"]:
        xpos = systematics["xpos"]
        # Smooth the positions in case the locators had trouble.
        xpos = median_timeseries_filter(xpos,sigma=3.0,kernel=21)
    
    if systematics["spatial_detrend"]:
        ypos = systematics["ypos"]
        # Smooth the positions in case the locators had trouble.
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
        ld["ld_coeffs"] = ld["ld_initialguess"]
    
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