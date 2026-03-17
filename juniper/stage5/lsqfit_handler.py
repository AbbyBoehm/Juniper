import os
import time
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from juniper.stage5 import models

from juniper.stage5 import batman_handler, fit_handler, exotic_handler, models
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.cleaning import median_timeseries_filter
from juniper.util.plotting import plot_fit

def lsqfit(exp_times, light_curve, errors, wavelengths,
           planets, flares, systematics, ld,
           inpt_dict, is_spec=False,
           show_guess_plot=False, save_guess_plot=False,
           plot_dir=None, outfile=None, wavestr=None):
    """Performs linear least squares fitting on the given array(s) using scipy.
    
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
        
    # If you are doing a poly fit, use a quick numpy polyfit to improve the coefficient estimates.
    for i,key in enumerate(list(systematics.keys())):
        if systematics[key]["poly"]:
            poly_degree = len(systematics[key]["poly_coeffs"])-1
            '''
            # Assume event duration of 15% of the exposure duration.
            duration = 0.15*(exp_times[i][-1]-exp_times[i][0])

            # Identify flux away from the expected mid-transit and mid-eclipse times.
            oks = []
            for j, planet_key in enumerate(list(planets[key].keys())):
                planet = planets[key][planet_key]
                tkeys = (f't_prim{j+1}',f't_seco{j+1}')
                for tkey in tkeys:
                    event_time = planet[tkey]
                    not_ok = (exp_times[i]>event_time-duration) & (exp_times[i]<event_time+duration)
                    oks.append(~not_ok)
            all_ok = np.full(oks[0].shape,True)
            for ok in oks:
                all_ok = np.logical_and(all_ok,ok)

            polyfit_coeffs = np.flip(np.polyfit(exp_times[i][all_ok]-exp_times[i][0],
                                                light_curve[i][all_ok],
                                                deg=poly_degree))
            '''

            # Estimate transit model with just planets + depths
            planets[key] = batman_handler.batman_init_all_planets(exp_times[i], planets[key], ld[key],
                                                                  event=inpt_dict["event_type_"+key])
            faux_sys = {}
            for sys_key in list(systematics[key].keys()):
                faux_sys[sys_key] = False
            planet_flux, _ = models.full_model(exp_times[i],planets[key],
                                               {},faux_sys,None,None)

            polyfit_coeffs = np.flip(np.polyfit(exp_times[i]-exp_times[i][0],
                                                light_curve[i]/planet_flux,
                                                deg=poly_degree))

            systematics[key]["poly_coeffs"] = polyfit_coeffs

            # If asked, plot how we got the poly model.
            if (save_guess_plot or show_guess_plot):
                fig, ax = plt.subplots(figsize=(7,5))
                ax.scatter(exp_times[i],light_curve[i],color='k')
                #ax.scatter(exp_times[i][~all_ok],light_curve[i][~all_ok],color='grey')
                ax.scatter(exp_times[i],light_curve[i]/planet_flux,color='grey')
                poly_flux = models.systematic_polynomial(exp_times[i],polyfit_coeffs)
                ax.plot(exp_times[i],poly_flux,color='red')
                ax.set_xlabel("Exposure Time [BJD_TDB]")
                ax.set_ylabel("Flux [normalized]")
                ax.tick_params(which='both',axis='both',direction='in',)
                if save_guess_plot:
                    if is_spec:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"detector{}".format(key)+"_spec{}LSQ_system-estimate.png".format(wavestr)),
                                    dpi=300, bbox_inches='tight')
                    else:
                        plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"detector{}".format(key)+"_broadbandLSQ_system-estimate.png"),
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
            # We need to update our parameters then.
            ld[key]["ld_initialguess"] = exotic_handler.get_exotic_coefficients(ld[key])
            ld[key]["ld_coeffs"] = ld[key]["ld_initialguess"]

    # (Re-)Initialize the planets, giving them the ld info they need to talk to batman properly.
    for i,key in enumerate(list(planets.keys())):
        planets[key] = batman_handler.batman_init_all_planets(exp_times[i], planets[key], ld[key],
                                                              event=inpt_dict["event_type_"+key])

    # Build priors dictionaries, and log information about what is getting fit.
    param_priors, priors_types, fit_or_not = fit_handler.build_priors_dict(planets,flares,systematics,ld,
                                                                           is_spec=is_spec)
    
    # Then build the lsq bounds object.
    bounds = fit_handler.build_bounds(param_priors, priors_types)

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
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}LSQ-initial.png".format(wavestr)),
                            dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandLSQ-initial.png"),
                            dpi=300, bbox_inches='tight')
        if show_guess_plot:
            plt.show(block=True)
        plt.close()
    
    # Now do lsq.
    results = minimize(fit_handler._residuals,
                       x0=params_array,
                       args=(exp_times, light_curve, errors, bundled_params, fit_or_not,
                             preserve_timing, preserve_depth, preserve_orbit),
                       method=inpt_dict["LSQ_type"],
                       tol=inpt_dict["LSQ_tolerance"],
                       bounds=bounds,
                       options={"maxiter":inpt_dict["LSQ_iter"],
                                "disp":(inpt_dict["verbose"]==2)})
    
    if inpt_dict["verbose"] == 2:
        print(results.message)
    
    # The array is here.
    fitted_array = results.x

    # Turn it back into a dict.
    fitted_dict = fit_handler.array_to_dict(fitted_array, bundled_params, fit_or_not)

    # Repack the dict back into planets, flares, systematics, and lds.
    planets, flares, systematics, ld = {}, {}, {}, {}
    for key in list(fitted_dict.keys()):
        planets[key], flares[key], systematics[key], ld[key] = fit_handler.unpack_params_back_to_dicts(fitted_dict[key],
                                                                                                       originals[key])
        
    # Repeat preserve calls.
    planets = fit_handler.preservation(planets,
                                       preserve_timing, preserve_depth, preserve_orbit)

    # Re-initialize the planets to get the updated models into place.
    for i,key in enumerate(list(planets.keys())):
        planets[key] = batman_handler.batman_init_all_planets(exp_times[i], planets[key], ld[key],
                                                              event=inpt_dict["event_type_"+key])
        
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
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}LSQ-final.png".format(wavestr)),
                            dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandLSQ-final.png"),
                            dpi=300, bbox_inches='tight')
        if show_guess_plot:
            plt.show(block=True)
        plt.close()
    
    # And return the fitted parameters.
    return planets, flares, systematics, ld

def lsqfit_one(lc_time, light_curve, errors, waves, planets, flares, systematics, ld, inpt_dict, is_spec=False):
    """Performs linear least squares fitting on the given array(s) using scipy.
    Fits a single light curve. Useful for fitting spectroscopic curves.

    Args:
        lc_time (np.array): mid-exposure times for each point in the light curve.
        light_curve (np.array): median-normalized flux with time.
        errors (np.array): uncertainties associated with each data point, used
        in weighting the residuals.
        waves (np.array): used to supply the wavelength range to
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
        widths = median_timeseries_filter(widths,sigma=3.0,kernel=31)

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

    # Then build the lsq bounds object.
    bounds = fit_handler.build_bounds(params_priors, priors_type=inpt_dict["priors_type"])

    # Conveniently, the priors also tells us which keys are getting fit.
    fit_param_keys = list(params_priors.keys())

    # Translate planets, flares, systematics, and lds into a single fitting dictionary.
    params_to_fit = fit_handler.bundle_planets_flares_systematics_and_ld(planets,
                                                                         flares,
                                                                         systematics,
                                                                         ld)
    
    # Turn that into an array so scipy will accept it.
    params_array = fit_handler.dict_to_array(params_to_fit, fit_param_keys)

    # Now do lsq.
    results = minimize(fit_handler._residuals,
                       x0=params_array,
                       args=(lc_time, light_curve, errors, params_to_fit, fit_param_keys, xpos, ypos, widths),
                       method=inpt_dict["LSQ_type"],
                       tol=inpt_dict["LSQ_tolerance"],
                       bounds=bounds,
                       options={"maxiter":inpt_dict["LSQ_iter"]})
    
    if inpt_dict["verbose"] == 2:
        print(results.message)
    
    # The array is here.
    fitted_array = results.x

    # Turn it back into a dict.
    fitted_dict = fit_handler.array_to_dict(fitted_array, params_to_fit, fit_param_keys)

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
    
    # Fill in anything that went missing.
    planets, flares, systematics, ld = fit_handler.refill(planets,flares,systematics,ld,
                                                          old_planets,old_flares,old_systematics,old_ld)
    
    # Re-initialize the planets.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, ld,
                                                     event=inpt_dict["event_type"])
    
    # And return the fitted parameters.
    return planets, flares, systematics, ld

def lsqfit_joint(lc_time, light_curve, errors, waves, planets, flares, systematics, ld, inpt_dict, is_spec=False):
    """Performs linear least squares fitting on the given array(s) using scipy.
    Fits multiple light curves simultaneously, optionally forcing them to share
    parameters (i.e. timing parameters, depths, etc.).

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
        widths = median_timeseries_filter(widths,sigma=3.0,kernel=31)

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

    # Then build the lsq bounds object.
    bounds = fit_handler.build_bounds(params_priors, priors_type=inpt_dict["priors_type"])

    # Conveniently, the priors also tells us which keys are getting fit.
    fit_param_keys = list(params_priors.keys())

    # Translate planets, flares, systematics, and lds into a single fitting dictionary.
    params_to_fit = fit_handler.bundle_planets_flares_systematics_and_ld(planets,
                                                                         flares,
                                                                         systematics,
                                                                         ld)
    
    # Turn that into an array so scipy will accept it.
    params_array = fit_handler.dict_to_array(params_to_fit, fit_param_keys)

    # Now do lsq.
    results = minimize(fit_handler._residuals,
                       x0=params_array,
                       args=(lc_time, light_curve, errors, params_to_fit, fit_param_keys, xpos, ypos, widths),
                       method=inpt_dict["LSQ_type"],
                       tol=inpt_dict["LSQ_tolerance"],
                       bounds=bounds,
                       options={"maxiter":inpt_dict["LSQ_iter"]})
    
    if inpt_dict["verbose"] == 2:
        print(results.message)
    
    # The array is here.
    fitted_array = results.x

    # Turn it back into a dict.
    fitted_dict = fit_handler.array_to_dict(fitted_array, params_to_fit, fit_param_keys)

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
    
    # Fill in anything that went missing.
    planets, flares, systematics, ld = fit_handler.refill(planets,flares,systematics,ld,
                                                          old_planets,old_flares,old_systematics,old_ld)
    
    # Re-initialize the planets.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, ld,
                                                     event=inpt_dict["event_type"])
    
    # And return the fitted parameters.
    return planets, flares, systematics, ld
    
    
    
    
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
    if systematics["width_detrend"]:
        widths = systematics["width"]

    # Check if ExoTiC-LD is being used.
    if ld["use_exotic"]:
        # We need to update our parameters then.
        ld["wavelength_range"] = np.array([np.min(waves), np.max(waves)])
        ld["ld_initialguess"] = exotic_handler.get_exotic_coefficients(ld)

    # Make dictionaries for each detector, forcing all but rp, ld, and
    # systematics to be shared.
    detectors = {}
    for l in range(light_curve.shape[0]):
        key = "detector" + str(l+1)
        detector = {}
        detector["planets"] = planets
        detector["flares"] = flares
        detector["systematics"] = systematics
        detector["ld"] = ld
        detectors[key] = detector
    
    # (Re-)Initialize the planets in each detector, giving them the ld info they
    # need to talk to batman properly.
    for detector in detectors.keys():
        D = detectors[detector]
        D["planets"] = batman_handler.batman_init_all_planets(lc_time, D["planets"], D["ld"],
                                                              event=inpt_dict["event_type"])
    
        # Build a priors dictionary.
        D["priors"] = fit_handler.build_priors_dict(D["planets"],D["flares"],D["systematics"],D["ld"])

        # Then build the lsq bounds object.
        D["bounds"] = fit_handler.build_bounds(D["priors"], priors_type=inpt_dict["priors_type"])

        # Conveniently, the priors also tells us which keys are getting fit.
        D["fit_param_keys"] = list(D["priors"].keys())

        # Translate planets, flares, systematics, and lds into a single fitting dictionary.
        D["params_to_fit"] = fit_handler.bundle_planets_flares_systematics_and_ld(D["planets"],
                                                                                  D["flares"],
                                                                                  D["systematics"],
                                                                                  D["ld"])
    
        # Turn that into an array so scipy will accept it.
        D["params_array"] = fit_handler.dict_to_array(D["params_to_fit"], D["fit_param_keys"])

    # Now that we have N_detectors worth of params_arrays, we must consolidate them.
    params_array = fit_handler.consolidate_multiple_detectors(detectors)

    # Now do lsq.
    '''
    results = minimize(fit_handler._residuals,
                       x0=params_array,
                       args=(lc_time, light_curve, errors, params_to_fit, fit_param_keys, xpos, ypos, widths),
                       method=inpt_dict["LSQ_type"],
                       bounds=bounds)
    
    # The array is here.
    fitted_array = results.x

    # Turn it back into a dict.
    fitted_dict = fit_handler.array_to_dict(fitted_array, params_to_fit, fit_param_keys)

    # And then turn those back into planets, flares, and systematics.
    planets, flares, systematics, ld = fit_handler.unpack_params_back_to_dicts(fitted_dict,
                                                                               xpos,
                                                                               ypos,
                                                                               widths)
    
    # Fill in anything that went missing.
    planets, flares, systematics, ld = fit_handler.refill(planets,flares,systematics,ld,
                                                          old_planets,old_flares,old_systematics,old_ld)
    
    # Re-initialize the planets.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, ld,
                                                     event=inpt_dict["event_type"])
    '''
    # WIP!
    # And return the fitted parameters.
    return planets, flares, systematics, ld