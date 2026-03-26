import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import numpy as np
import dynesty
import matplotlib.pyplot as plt

from juniper.stage5 import batman_handler, fit_handler, exotic_handler, models
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.cleaning import median_timeseries_filter
from juniper.util.plotting import plot_fit


def nestfit(exp_times, light_curve, errors, wavelengths,
            planets, flares, systematics, ld,
            inpt_dict, is_spec=False,
            show_guess_plot=False, save_guess_plot=False,
            plot_dir=None, outfile=None, wavestr=None):
    """Performs static or dynamic nested sampling fitting on the given array(s) using dynesty.
    
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
        show_guess_plot (bool, optional): whether to show a plot of the initial
        vs final guess. Defaults to False.
        save_guess_plot (bool, optional): whether to save a plot of the initial
        vs final guess. Defaults to False.
        plot_dir (str, optional): location to save diagnostic plots to. Defaults to None.
        outfile (str, optional): name to save diagnostic plots to. Defaults to None.
        wavestr (str, optional): name to save spec plots to. Defaults to None.
    
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
        
    # Apply log-uniform transforms as needed.
    planets, flares, ld = fit_handler.loguniform_transform(planets, flares, ld)
    
    # If you are doing a poly fit, use a quick numpy polyfit to improve the coefficient estimates.
    for i,key in enumerate(list(systematics.keys())):
        if systematics[key]["poly"]:
            poly_degree = len(systematics[key]["poly_coeffs"])-1

            # Estimate transit model with just planets + flares    
            planets[key] = batman_handler.batman_init_all_planets(exp_times[i], planets[key], ld[key],
                                                                  event=inpt_dict["event_type_"+key])
            faux_sys = {}
            for sys_key in list(systematics[key].keys()):
                faux_sys[sys_key] = False
            
            planet_flux, _ = models.full_model(exp_times[i],planets[key],
                                               flares[key],faux_sys,None,None)
            polyfit_coeffs = np.flip(np.polyfit(exp_times[i]-exp_times[i][0],
                                                (light_curve[i]/planet_flux)/np.median(light_curve[i]),
                                                deg=poly_degree))
            
            systematics[key]["poly_coeffs"] = polyfit_coeffs
            polyfit_coeffs[0] = np.median(light_curve[i])*polyfit_coeffs[0]

            # If asked, plot how we got the poly model.
            if (save_guess_plot or show_guess_plot):
                fig, ax = plt.subplots(figsize=(7,5))
                ax.scatter(exp_times[i],light_curve[i],color='k')
                ax.scatter(exp_times[i],light_curve[i]/planet_flux,color='grey')
                poly_flux = models.systematic_polynomial(exp_times[i],polyfit_coeffs)
                ax.plot(exp_times[i],poly_flux,color='red')
                ax.set_xlabel("Exposure Time [BJD_TDB]")
                ax.set_ylabel("Flux [normalized]")
                ax.tick_params(which='both',axis='both',direction='in',)
                if save_guess_plot:
                    if is_spec:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"detector{}".format(key)+"_spec{}nested_system-estimate.png".format(wavestr)),
                                    dpi=300, bbox_inches='tight')
                    else:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"detector{}".format(key)+"_broadbandnested_system-estimate.png"),
                                    dpi=300, bbox_inches='tight')
                if show_guess_plot:
                    plt.show(block=True)
                plt.close()

            if (save_guess_plot or show_guess_plot):
                fig, ax = plt.subplots(figsize=(7,5))
                ax.scatter(exp_times[i],light_curve[i]/np.median(light_curve[i]),color='grey',zorder=0)
                ax.plot(exp_times[i],planet_flux,color='k',zorder=1)
                ax.set_xlabel("Exposure Time [BJD_TDB]")
                ax.set_ylabel("Flux [normalized]")
                ax.tick_params(which='both',axis='both',direction='in',)
                if save_guess_plot:
                    if is_spec:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"detector{}".format(key)+"_spec{}nested_system-planetflare.png".format(wavestr)),
                                    dpi=300, bbox_inches='tight')
                    else:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"detector{}".format(key)+"_broadbandnested_system-planetflare.png"),
                                    dpi=300, bbox_inches='tight')
                if show_guess_plot:
                    plt.show(block=True)
                plt.close()
    
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
    param_priors, priors_types, fit_or_not = fit_handler.build_priors_dict(planets,flares,systematics,ld,
                                                                           is_spec=is_spec)
    
    # Unpack priors into a list of lists.
    unpack_priors = []
    for superdict_key in list(planets.keys()):
        for key in list(param_priors[superdict_key].keys()):
            unpack_priors.append(param_priors[superdict_key][key])

    # Unpack prior types into a list of lists.
    unpack_ptypes = []
    for superdict_key in list(planets.keys()):
        for key in list(priors_types[superdict_key].keys()):
            unpack_ptypes.append(priors_types[superdict_key][key])

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
    preserve_star = inpt_dict["preserve_star"]

    # If asked, make a plot of the initial guess.
    if (show_guess_plot or save_guess_plot):
        fig, ax = plt.subplots(figsize=(7,int(2.5*len(list(planets.keys())))),
                               nrows=len(list(planets.keys())))
        # On each ax[i], plot the full model and its components.
        for i,key in enumerate(list(planets.keys())):
            full_model, _ = models.full_model(exp_times[i],
                                              planets[key],
                                              flares[key],
                                              systematics[key],
                                              None, None)
            
            # Plot the initial model over the data.
            if len(list(planets.keys())) > 1:
                ax[i] = plot_fit(ax[i], exp_times[i], light_curve[i], errors[i],
                                exp_times[i], full_model, fit_color='blue')
                ax[i].set_xlabel("Exposure Time [BJD_TDB]")
                ax[i].set_ylabel("Flux [normalized]")
                ax[i].tick_params(which='both',axis='both',direction='in',)
            else:
                ax = plot_fit(ax, exp_times[i], light_curve[i], errors[i],
                                exp_times[i], full_model, fit_color='blue')
                ax.set_xlabel("Exposure Time [BJD_TDB]")
                ax.set_ylabel("Flux [normalized]")
                ax.tick_params(which='both',axis='both',direction='in',)
        if save_guess_plot:
            if is_spec:
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}nested-initial.png".format(wavestr)),
                            dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandnested-initial.png"),
                            dpi=300, bbox_inches='tight')
        if show_guess_plot:
            plt.show(block=True)
        plt.close()

    # Define dimensionality of the problem.
    ndim = params_array.shape[0]

    # Define args of the log probability function.
    loglike_args = (bundled_params, fit_or_not, exp_times, light_curve, errors,
                    unpack_priors, unpack_ptypes,
                    preserve_timing, preserve_depth, preserve_orbit, preserve_star)
    
    # Define args of the prior transform function.
    ptform_args = (param_priors, priors_types)

    # Establish dlnz.
    dlnz = None
    if "nested_dlnz" in list(inpt_dict.keys()):
        dlnz = inpt_dict["nested_dlnz"]

    # Check for parallelization as we initialize and run the sampler.
    if inpt_dict["max_cores"] not in (1,'1'):
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
        with dynesty.pool.Pool(n_use,fit_handler.log_probability,fit_handler.prior_transform,
                               logl_args=loglike_args,ptform_args=ptform_args) as dypool:
            # Create the sampler object with a pool.
            if inpt_dict["nested_dynamic"]:
                # Use a dynamic sampler.
                sampler = dynesty.DynamicNestedSampler(dypool.loglike,
                                                       dypool.prior_transform,
                                                       ndim,
                                                       bound = inpt_dict["nested_bounding"],
                                                       sample = inpt_dict["nested_sample"],
                                                       pool=dypool)
                if inpt_dict["verbose"] == 2:
                    print("Beginning dynamic nested sampling...")
                sampler.run_nested(dlogz_init=dlnz,
                                   nlive_init=inpt_dict["nested_nlive"],
                                   nlive_batch=inpt_dict["nested_batch"])
            else:
                # Use a static sampler.
                sampler = dynesty.NestedSampler(dypool.loglike,
                                                dypool.prior_transform,
                                                ndim,
                                                nlive = inpt_dict["nested_nlive"],
                                                bound = inpt_dict["nested_bounding"],
                                                sample = inpt_dict["nested_sample"],
                                                pool=dypool)
                if inpt_dict["verbose"] == 2:
                    print("Beginning static nested sampling...")
                sampler.run_nested(dlogz=dlnz,)
    else:
        # Create the sampler object without a pool.
        print("Running Dynesty without multiprocessing. This might take awhile...")

        # Create the nested sampler object.
        if inpt_dict["nested_dynamic"]:
            # Use a dynamic sampler.
            sampler = dynesty.DynamicNestedSampler(fit_handler.log_probability,
                                                   fit_handler.prior_transform,
                                                   ndim,
                                                   bound = inpt_dict["nested_bounding"],
                                                   sample = inpt_dict["nested_sample"],
                                                   logl_args=loglike_args,
                                                   ptform_args=ptform_args,)
            if inpt_dict["verbose"] == 2:
                print("Beginning dynamic nested sampling...")
            sampler.run_nested(dlogz_init=dlnz,
                               nlive_init=inpt_dict["nested_nlive"],
                               nlive_batch=inpt_dict["nested_batch"])
        else:
            # Use a static sampler.
            sampler = dynesty.NestedSampler(fit_handler.log_probability,
                                            fit_handler.prior_transform,
                                            ndim,
                                            nlive = inpt_dict["nested_nlive"],
                                            bound = inpt_dict["nested_bounding"],
                                            sample = inpt_dict["nested_sample"],
                                            logl_args=loglike_args,
                                            ptform_args=ptform_args,)
            if inpt_dict["verbose"] == 2:
                print("Beginning static nested sampling...")
            sampler.run_nested(dlogz=dlnz,)

    # Get the results from the sampler.
    results = sampler.results

    # Print a summary of the run if asked.
    if inpt_dict["verbose"] == 2:
        print(results.summary)

    # Determine how many steps are burn-in.
    if inpt_dict["nested_burnin"] > 0:
        discard = int(inpt_dict["nested_burnin"]*results.niter)
    else:
        discard = 0
    
    # Pull the sampled posteriors and discard the burn-in.
    samples = results.samples[discard:]

    # Turn the flattened chains into arrays.
    fitted_array, fitted_errs_array = fit_handler.get_result_from_post(ndim, samples)

    # Turn both back into dicts.
    fitted_dict = fit_handler.array_to_dict(fitted_array, bundled_params, fit_or_not)
    fitted_errs_dict = fit_handler.array_to_dict(fitted_errs_array, bundled_params, fit_or_not)

    # Repack the dict back into planets, flares, systematics, and lds.
    planets, flares, systematics, ld = {}, {}, {}, {}
    for key in list(fitted_dict.keys()):
        planets[key], flares[key], systematics[key], ld[key] = fit_handler.unpack_params_back_to_dicts(fitted_dict[key],
                                                                                                       originals[key])
    
    # Apply log-uniform transforms as needed.
    planets, flares, ld = fit_handler.loguniform_transform(planets, flares, ld)

    # Repeat preserve calls.
    planets, ld = fit_handler.preservation(planets, ld,
                                           preserve_timing, preserve_depth,
                                           preserve_orbit, preserve_star)

    # Same for errors.
    planets_err, flares_err, systematics_err, ld_err = {}, {}, {}, {}
    for key in list(fitted_errs_dict.keys()):
        planets_err[key], flares_err[key], systematics_err[key], ld_err[key] = fit_handler.unpack_params_back_to_dicts(fitted_errs_dict[key],
                                                                                                                       originals[key])
    
    # Apply log-uniform transforms as needed.
    planets_err, flares_err, ld_err = fit_handler.loguniform_transform(planets_err, flares_err, ld_err)
    
    # Repeat preserve calls.
    planets_err, ld_err = fit_handler.preservation(planets_err, ld_err,
                                                   preserve_timing, preserve_depth,
                                                   preserve_orbit, preserve_star)

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
    if preserve_star:
        # Every spare copy of ldN must be deleted.
        delete_parameter = "ld"
        for ldN in range(1,10):
            parameter_indices = [i for i, x in enumerate(labels) if '{}{}'.format(delete_parameter,ldN) == x]
            for i in range(1,len(parameter_indices)): # the 1 lets us skip the first instance
                delete_indices.append(parameter_indices[i])
    
    # Now delete the indices, if any were found.
    if delete_indices:
        if inpt_dict["verbose"] == 2:
            print("Found {} repeat indices to delete as part of preserve arguments.".format(len(delete_indices)))
        delete_indices = np.array([int(i) for i in delete_indices])
        samples = np.delete(samples,obj=delete_indices,axis=1)
        labels = np.delete(labels,obj=delete_indices,axis=0)
    
    # Update ndim, necessary if there were deletions.
    ndim = samples.shape[1]

    # Store plotting items, we may want them later.
    plotting_items = (ndim, samples, labels)

    # If asked, make a plot of the final guess.
    if (show_guess_plot or save_guess_plot):
        fig, ax = plt.subplots(figsize=(7,int(2.5*len(list(planets.keys())))),
                               nrows=len(list(planets.keys())))
        # On each ax[i], plot the full model and its components.
        for i,key in enumerate(list(planets.keys())):
            full_model, _ = models.full_model(exp_times[i],
                                              planets[key],
                                              flares[key],
                                              systematics[key],
                                              None, None)
            
            # Plot the final model over the data.
            if len(list(planets.keys())) > 1:
                ax[i] = plot_fit(ax[i], exp_times[i], light_curve[i], errors[i],
                                exp_times[i], full_model, fit_color='red')
                ax[i].set_xlabel("Exposure Time [BJD_TDB]")
                ax[i].set_ylabel("Flux [normalized]")
                ax[i].tick_params(which='both',axis='both',direction='in',)
            else:
                ax = plot_fit(ax, exp_times[i], light_curve[i], errors[i],
                                exp_times[i], full_model, fit_color='red')
                ax.set_xlabel("Exposure Time [BJD_TDB]")
                ax.set_ylabel("Flux [normalized]")
                ax.tick_params(which='both',axis='both',direction='in',)
        if save_guess_plot:
            if is_spec:
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}nested-final.png".format(wavestr)),
                            dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandnested-final.png"),
                            dpi=300, bbox_inches='tight')
        if show_guess_plot:
            plt.show(block=True)
        plt.close()
    
    # And return the fitted parameters.
    return planets, flares, systematics, ld, planets_err, flares_err, systematics_err, ld_err, plotting_items
