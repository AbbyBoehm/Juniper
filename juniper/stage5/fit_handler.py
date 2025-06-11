import numpy as np

from juniper.stage5 import models

def bundle_planets_flares_systematics_and_ld(planets,flares,systematics,ld):
    """Simple function which unpacks every provided planet, flare, systematics
    model, and limb darkening model and puts them into a single dictionary to
    be passed to least-squares and emcee fitters.

    Args:
        planets (dict): series of entries describing each planet in the model,
        tagged by "planet1", "planet2", etc.
        flares (dict): series of entries describing each flare in the model,
        tagged by "flare1", "flare2", etc.
        systematics (dict): systematics info containing "poly", "poly_coeffs",
        "mirrortilt", "mirrortilt_coeffs", etc.
        ld (dict): information on stellar limb darkening model. If you are fitting
        for this, you need to add its info in.

    Returns:
        dict: contents of all four dictionaries spilled into an array.
    """
    bundled_params = {}
    for planet_name in planets.keys():
        planet = planets[planet_name]
        for key in planet.keys():
            if "prior" in key:
                continue
            bundled_params[key] = planet[key]
    for flare_ID in flares.keys():
        flare = flares[flare_ID]
        for key in flare.keys():
            if "prior" in key:
                continue
            bundled_params[key] = flare[key]
    for key in systematics.keys():
        if all(special_key not in key for special_key in ("coeffs","xpos","ypos","width")):
            continue
        if (key == "disp_detrend" or key == "spatial_detrend" or key == "width_detrend"):
            continue
        bundled_params[key] = systematics[key]
    
    # Whether we are fitting or not, we need the ld info.
    special_keys = ["ld_model","fit_lds","ld_initialguess","ld_coeffs","use_exotic","ld_data_path","ld_grid",
                    "custom_grid","interpolate","instrument_mode","stellar_params","wavelength_range",]
    for key in special_keys:
        bundled_params[key] = ld[key]

    return bundled_params

def unpack_params_back_to_dicts(bundled_params, originals=None):
    """Slightly less simple function which takes the unified parameters
    dictionary and separates it back into planets, flares, and systematics.
    Also refills the prior info.

    Args:
        bundled_params (dict): contains keys like "rp1", "A1", and "poly_coeffs".
        originals (dict or None): contains "planets", "systematics", etc. keys and
        retains priors and bools that are not kept in bundled_params.

    Returns:
        dict, dict, dict, dict: the planets, flares, systematics and ld dictionaries rebuilt.
    """
    # First, find every planet.
    special_keys = ["rp","fp","t_prim","t_seco","period","aor","incl","ecc",
                    "longitude","batman_model","batman_params"]
    # Now we are going to parse params_to_fit for individual planets.
    planets = {}
    planet_N = 1 # we're going to step up planet_N until we run out of planets
    had_KeyError = False # haven't hit a key error
    while not had_KeyError:
        # Create a key for the Nth planet.
        planet_name = "planet{}".format(planet_N)
        # And a dictionary for that planet.
        planet = {}
        try:
            for key in special_keys:
                planet[key+str(planet_N)] = bundled_params[key+str(planet_N)] # as long as a planet of this number exists, params_to_fit will have this key
            
            # Load in the keys not there.
            if originals != None:
                for extra_key in [k for k in list(originals['planets'][planet_name].keys()) if k not in list(planet.keys())]:
                    planet[extra_key] = originals['planets'][planet_name][extra_key]
            
            # Now add the planet to the planets.
            planets[planet_name] = planet

            # And step up the planet number.
            planet_N += 1

        except KeyError:
            # We have run out of planet.
            had_KeyError = True
    
    # We found all the planets. Now to find all the flares.
    special_keys = ["A","B","C","Dr","Ds","Fr","E"]

    # Now we are going to parse params_to_fit for individual flares.
    flares = {}
    flare_ID = 1 # we're going to step up flare_ID until we run out of flares
    had_KeyError = False # haven't hit a key error
    while not had_KeyError:
        # Create a key for the next flare.
        flare_name = "flare{}".format(flare_ID)
        # And a dictionary for that flare.
        flare = {}
        try:
            for key in special_keys:
                flare[key+str(flare_ID)] = bundled_params[key+str(flare_ID)] # as long as a flare of this number exists, params_to_fit will have this key
            
            # Load in the keys not there.
            if originals != None:
                for extra_key in [k for k in list(originals['flares'][flare_ID].keys()) if k not in list(flare.keys())]:
                    flare[extra_key] = originals['flares'][flare_ID][extra_key]

            # Now add the flare to the flares.
            flares[flare_name] = flare

            # And step up the flare ID number.
            flare_ID += 1

        except KeyError:
            # We have run out of flare.
            had_KeyError = True
    
    # We found the planets and the flares. Now to parse the systematics.
    special_keys = ["poly","mirrortilt","disp_detrend","spatial_detrend","width_detrend","singleramp","doubleramp"]
    systematics = {}
    for key in special_keys:
        try:
            systematics[key+"_coeffs"] = bundled_params[key+"_coeffs"]
            systematics[key] = True # if we haven't failed yet, then this must be True.
            if key == "disp_detrend":
                pos_key = "xpos"
                systematics[pos_key] = bundled_params[pos_key]
            if key == "spatial_detrend":
                pos_key = "ypos"
                systematics[pos_key] = bundled_params[pos_key]
            if key == "width_detrend":
                pos_key = "width"
                systematics[pos_key] = bundled_params[pos_key]
        except:
            # If this failed, then we were not fitting that kind of systematic.
            systematics[key] = False
    
    # Load in the keys not there.
    if originals != None:
        for extra_key in [k for k in list(originals["systematics"].keys()) if k not in list(systematics.keys())]:
            systematics[extra_key] = originals["systematics"][extra_key]

    # Finally, we need to read the ld info back out.
    special_keys = ["ld_model","fit_lds","ld_initialguess","ld_coeffs","use_exotic","ld_data_path","ld_grid",
                    "custom_grid","interpolate","instrument_mode","stellar_params","wavelength_range",]
    ld = {}
    for key in special_keys:
        try:
            ld[key] = bundled_params[key]
        except:
            # We expect an exception if ld info was not fit to begin with.
            pass
    
    # Load in the keys not there.
    if originals != None:
        for extra_key in [k for k in list(originals["ld"].keys()) if k not in list(ld.keys())]:
            ld[extra_key] = originals["ld"][extra_key]

    # And we have now unpacked the params_to_fit dict.
    return planets, flares, systematics, ld

def refill(new_planets, new_flares, new_systematics, new_ld, old_planets, old_flares, old_systematics, old_ld):
    """Checks if the newly-fitted dictionaries are missing anything and replaces the missing entries.

    Args:
        new_planets (dict): series of entries describing the newly-fitted planets.
        new_flares (dict): series of entries describing the newly-fitted flares.
        new_systematics (dict): series of entries describing the newly-fitted systematics.
        new_ld (dict): series of entries describing the newly-fitted limb darkening.
        old_planets (dict): series of entries describing the original planets.
        old_flares (dict): series of entries describing the original flares.
        old_systematics (dict): series of entries describing the original systematics.
        old_ld (dict): series of entries describing the original limb darkening.

    Returns:
        dict, dict, dict: the new planets, flares, and systematics with any holes filled.
    """
    # Check if anything is missing from the new planets.
    for planet_name in new_planets.keys():
        new_planet = new_planets[planet_name]
        old_planet = old_planets[planet_name]
        unfilled_keys = [key for key in old_planet.keys() if key not in new_planet.keys()]
        for key in unfilled_keys:
            new_planet[key] = old_planet[key]
    
    # And the new flares.
    for flare_ID in new_flares.keys():
        new_flare = new_flares[flare_ID]
        old_flare = old_flares[flare_ID]
        unfilled_keys = [key for key in old_flare.keys() if key not in new_flare.keys()]
        for key in unfilled_keys:
            new_flare[key] = old_flare[key]

    # And the systematics.
    unfilled_keys = [key for key in old_systematics.keys() if key not in new_systematics.keys()]
    for key in unfilled_keys:
        new_systematics[key] = old_systematics[key]

    # And the lds.
    unfilled_keys = [key for key in old_ld.keys() if key not in new_ld.keys()]
    for key in unfilled_keys:
        new_ld[key] = old_ld[key]

    return new_planets, new_flares, new_systematics, new_ld

def dict_to_array(bundled_params, fit_or_not):
    """Simple function to take the dictionary of parameters that needs to be
    fitteed and spit out its contents as an array.

    Args:
        bundled_params (dict): all of the parameters needed to fit, including
        every rp1, rp2, rpN, plus every flare, systematic model, and lds.
        Each is stored in a '1', '2', etc. dict for each spectrum.
        fit_or_not (dict): all of the keys that exist in this fit for every
        spectrum, attached to a bool stating whether or not that parameter
        is fit.

    Returns:
        np.array, dict: an array of the contents of params_to_fit, and a guide
        to which parameters need to be updated.
    """
    # Let's set up the array that will soon hold everything. Start as a list.
    params_to_arrayify = []

    # We need to go through this entire process one superdict at a time.
    for superdict_key in list(bundled_params.keys()):
        # Each superdict_key has the form "1", "2", etc. and its contents
        # are otherwise as the single-spec dicts we have used before.
        
        # We start by fetching the correct parser.
        fit_param_keys = fit_or_not[superdict_key]

        # While bundled_params does contain everything needed to evaluate a fit, not
        # all of its contents are tunables. Plus, some of its contents are lists that
        # must be pulled apart. So let's crack into it.
        params_in_spec = bundled_params[superdict_key]

        # First, copy wholesale what can easily be copied.
        for key in list(fit_param_keys.keys()):
            # We can quickly skip anything that is not in fit_param_keys.
            if fit_param_keys[key]:
                # So the bool was True, that means it must be fit for.
                key = str.replace(key,"_prior","") # clean up the prior tag
                try:
                    if isinstance(params_in_spec[key],float) or isinstance(params_in_spec[key],int):
                        # It's a simple float or integer, so we can just tack it on there.
                        params_to_arrayify.append(params_in_spec[key])
                except KeyError:
                    # This key does not exist in params_to_fit, so it must be a special key (ld, poly, etc.).
                    pass

        # Now parse the systematics.
        system_keys = ["poly","mirrortilt","disp_detrend","spatial_detrend","width_detrend","singleramp","doubleramp"]
        for key in system_keys:
            # There will always be at least coeff1 in any model. So this is a simple
            # way to check that this model is being fitted. If it is being fit, it is True.
            try:
                if fit_param_keys[key+str(1)]:
                    key = str.replace(key,"_prior","") # clean up the prior tag
                    coeffs = params_in_spec[key+"_coeffs"] # all of the coefficients are bundled here.
                    for coeff in coeffs:
                        params_to_arrayify.append(coeff)
            except KeyError:
                # This model is not being fitted so this key does not exist.
                pass
        
        # Now check out lds.
        try:
            for i, bool in enumerate(params_in_spec["fit_lds"]):
                # For every bool here, check if it's fitted.
                if bool:
                    params_to_arrayify.append(params_in_spec["ld_coeffs"][i])
        except KeyError:
            # We are not fitting lds at all, so pass.
            pass

    # Make it an array! Now we can give it to scipy.
    arr = np.array(params_to_arrayify)
    return arr

def array_to_dict(params_array, input_param_dict, fit_or_not):
    """Slightly less simple function which uses the fit_or_not dict and the
    array-ified params_array to re-dictionary-ify the array, or to un-array-ify
    the array back into the dictionary it began as. This is a headache, haha!

    Args:
        params_array (np.array): the array-ified version of the fitted model.
        fit_or_not (dict): a dictionary of every single parameter for every
        input_param_dict (dict): the original dictionary that was given to the
        fitter when the models.full_model() was initialized.
        single spectrum recording fit instructions.

        params_to_fit (dict): the original dictionary that was given to the
        fitter when the models.full_model() was initialized.
        fit_param_keys (list of str): the keys that are actually getting
        modified during fitting.

    Returns:
        dict: the parameters back in dictionary form.
    """
    # We open a new dictionary.
    redicted_params = {}

    # We define some keys as requiring special treatment.
    special_keys = ["poly","mirrortilt","disp_detrend","spatial_detrend","width_detrend","singleramp","doubleramp"]

    # We track the params_array index.
    i = 0

    # Recall that the original bundled_params went by superdict, so
    # we will have to go by superdict as well.
    for superdict_key in list(fit_or_not.keys()):
        # Let's re-open the parameter dictionary for this spectrum.
        redicted_params[superdict_key] = {}

        # Track special keys.
        special_key_issues = {}
        lds = []

        # So the contents of each bundled_params were not separated. Thus,
        # if we just parse the fit_or_not[superdict_key] information in order,
        # it should restore the order of the keys naturally.
        for key in list(fit_or_not[superdict_key].keys()):
            # Check if that key was fitted.
            if fit_or_not[superdict_key][key]:
                # It was fitted, so pull from params_array.
                key = str.replace(key,"_prior","") # clean up the prior tag

                # Check if it is a special key.
                if any(special in key for special in special_keys):
                    # Will handle these later, they need repacked properly.
                    special_key_issues[key] = params_array[i]
                elif "ld" in key:
                    # Will handle these later, they need repacked properly.
                    lds.append(params_array[i])
                else:
                    redicted_params[superdict_key][key] = params_array[i]
                i += 1
            else:
                # It was not fitted, pull from input_param_dict.
                key = str.replace(key,"_prior","") # clean up the prior tag
                redicted_params[superdict_key][key] = input_param_dict[superdict_key][key]
        
        # There will be some missing bat info.
        hadKeyError = False
        bat_index = 1
        while not hadKeyError:
            try:
                batman_model_key = 'batman_model{}'.format(bat_index)
                batman_param_key = 'batman_params{}'.format(bat_index)
                for key in (batman_model_key,batman_param_key):
                    redicted_params[superdict_key][key] = input_param_dict[superdict_key][key]
                bat_index += 1
            except KeyError:
                hadKeyError = True

        # Let's tango with the special keys.
        for special_key in special_keys:
            # We need to bundle these when they were found in special_key_issues.
            new_bundle_key = "{}_coeffs".format(special_key)
            bundle = []
            for key in list(special_key_issues.keys()):
                if special_key in key:
                    bundle.append(special_key_issues[key])
            # Add it into the redict only if it existed.
            if bundle:
                redicted_params[superdict_key][new_bundle_key] = bundle

            # Fetch some extra info.
            if special_key == "disp_detrend":
                # At this time, grab xpos.
                key = "xpos"
                redicted_params[superdict_key][key] = input_param_dict[superdict_key][key]
            if special_key == "spatial_detrend":
                # At this time, grab ypos.
                key = "ypos"
                redicted_params[superdict_key][key] = input_param_dict[superdict_key][key]
            if special_key == "width_detrend":
                # At this time, grab xpos and ypos.
                key = "width"
                redicted_params[superdict_key][key] = input_param_dict[superdict_key][key]

        # Finally, we need to pull the ld_initialguess and fit_lds info.
        ld_keys = ["ld_model","fit_lds","ld_initialguess","ld_coeffs","use_exotic","ld_data_path","ld_grid",
                   "custom_grid","interpolate","instrument_mode","stellar_params","wavelength_range",]
        for key in ld_keys:
            if key != "ld_coeffs":
                redicted_params[superdict_key][key] = input_param_dict[superdict_key][key]
            else:
                # Need a bit more nuanced handling again. Right now, lds has been populated only
                # with the lds that are getting fit. We need to check input_param_dict[superdict_key]['fit_lds']
                # When we find True, we grab a value from lds.
                # When we find False, we grab a value from input_param_dict[superdict_key]['ld_initialguess'].
                lds_updated = []
                lds_index = 0
                for j, bool in enumerate(input_param_dict[superdict_key]['fit_lds']):
                    if bool:
                        # It was fit; we take the next value of lds.
                        lds_updated.append(lds[lds_index])
                        lds_index += 1
                    else:
                        # It was not fit: we take the current index i of the initial guess.
                        lds_updated.append(input_param_dict[superdict_key]['ld_initialguess'][j])
                redicted_params[superdict_key][key] = lds_updated
    
    return redicted_params

    '''
    # Remove _prior from keys.
    fit_param_keys = [str.replace(key,"_prior","") for key in fit_param_keys]
    # Open a new dictionary.
    redicted_params = {}
    systematics = {}
    lds = {}

    # Stash initial ld guess.
    redicted_params["ld_initialguess"] = params_to_fit["ld_initialguess"]
    redicted_params["fit_lds"] = params_to_fit["fit_lds"]

    # Keep "organized keys" for later.
    organized_keys = list(params_to_fit.keys())

    # Define systematics keys.
    special_keys = ["poly","mirrortilt","disp_detrend","spatial_detrend","width_detrend","singleramp","doubleramp"]
    for i, key in enumerate(fit_param_keys):
        # Some of these keys will be able to be transferred wholesale.
        # The exceptions are systematics models (must be bundled as "model_coeffs")
        # and limb darkening coefficients (must be bundled as "ld_initialguess")
        if any([special_key in key for special_key in special_keys]):
            # It's a systematic moodel coefficient! Store it to process later.
            systematics[key+str(i)] = params_array[i]
        elif "ld" in key:
            # It's an ld! Store it to process later.
            lds[key] = params_array[i]
        else:
            # It's nothing special, just take it as is.
            redicted_params[key] = params_array[i]

    # Now we need to put the systematics back in.
    for special_key in special_keys:
        # The systematics are checked out one by one. poly, then mirrortilt, then etc.
        system_keys = [key for key in systematics.keys() if special_key in key]
        system_model = []
        for key in system_keys:
            system_model.append(systematics[key])
        # And now we bundle all the systematic coefficients together again.
        if system_model:
            # That is, only if that system model is there at all. No need to make empty tags.
            redicted_params[special_key+"_coeffs"] = system_model

    # And let's put the lds back in.
    for key in lds.keys():
        # The key itself will tell us redicted_params["ld_initialguess"] items to update.
        index_to_update = int(str.replace(key,"ld",""))-1
        redicted_params["ld_initialguess"][index_to_update] = lds[key]

    # Finally, fill in anything that is missing.
    for key in params_to_fit.keys():
        if key not in redicted_params.keys():
            # If it was not fitted for, resupply it here.
            redicted_params[key] = params_to_fit[key]

    # It is important that the order of items in the dictionary is correct.
    reorganized_params = {}
    for key in organized_keys:
        reorganized_params[key] = redicted_params[key]
    
    return reorganized_params
    '''

def build_priors_dict(planets, flares, systematics, ld, is_spec=False):
    """Simple function to get the priors on every fitting parameter.

    Args:
        planets (dict): series of entries describing each planet in the model.
        flares (dict): series of entries describing each flare in the model.
        systematics (dict): series of entries describing systematic trends
        in the model.
        ld (dict): series of entries describing the limb darkening model.
        is_spec (bool, optional): whether this is a fit to a spectroscopic
        curve, in which case certain system parameters are to be locked.
        Defaults to False.
    
    Returns:
        dict, dict: each entry is a list of two numbers and this dict will be fed
        into the build_bounds function. We also return a dict which tells us what
        is and is not fitted.
    """
    # Initialize the priors dict.
    param_priors = {}
    # Also start a dict of what is and is not fitted.
    fit_or_not = {}

    # We go in order of superdict, and stay inside a try until it breaks.
    superdict_keys = planets.keys()
    for superdict_key in superdict_keys:
        # Start up the dicts that will correspond to param_priors[superdict_key].
        superdict_prior = {}
        superdict_fitornot = {}

        # First, gut every planet inside the superdict.
        special_keys = ["rp","fp","t_prim","t_seco","period","aor","incl","ecc","longitude"]
        ban_keys = []
        if is_spec:
            # In spectroscopic fits, we only concern ourselves with depth.
            # Physical system parameters are not to be fit for, so we ban them.
            ban_keys = [key for key in special_keys if key not in ["rp","fp"]]
            ban_keys = [i+"_prior" for i in ban_keys]
        special_keys = [i+"_prior" for i in special_keys]

        # Parse the planets only within the correct superdict entry.
        for i, planet_name in enumerate(planets[superdict_key].keys()):
            # Retrieve the right planet to look at.
            planet = planets[superdict_key][planet_name]
            
            # We have to make some truncations when dealing with specs.
            for key in ban_keys:
                planet[key+str(i+1)] = None                

            for key in special_keys:
                if planet[key+str(i+1)]: # if this is not None, it's being fitted.
                    superdict_prior[key+str(i+1)] = planet[key+str(i+1)]
                    superdict_fitornot[key+str(i+1)] = True
                else:
                    superdict_fitornot[key+str(i+1)] = False
    
        # We gutted all the planets. Now to gut all the flares.
        special_keys = ["A","B","C","Dr","Ds","Fr","E"]
        special_keys = [i+"_prior" for i in special_keys]

        # Parse the flares only within the correct superdict entry.        
        for i, flare_ID in enumerate(flares[superdict_key].keys()):
            flare = flares[superdict_key][flare_ID]
            for key in special_keys:
                if flare[key+str(i+1)]: # if this is not None, it's being fitted.
                    superdict_prior[key+str(i+1)] = flare[key+str(i+1)]
                    superdict_fitornot[key+str(i+1)] = True
                else:
                    superdict_fitornot[key+str(i+1)] = False
            
        # We need to unpack systematic info.
        special_keys = ["poly","mirrortilt","disp_detrend","spatial_detrend","width_detrend","singleramp","doubleramp"]
        for key in special_keys:
            if systematics[superdict_key][key]:
                # If this systematic is included, we need to put a wicked broad bound on every parameter.
                for i,coeff in enumerate(systematics[superdict_key][key+"_coeffs"]):
                    superdict_prior[key+str(i+1)] = [-1e40,1e40]
                    superdict_fitornot[key+str(i+1)] = True

        # And ld info, if applicable.
        for i, bool in enumerate(ld[superdict_key]["fit_lds"]):
            # If any of the lds are getting fit, we need a bound on it.
            if bool:
                superdict_prior["ld"+str(i+1)] = [-10,10]
                superdict_fitornot["ld"+str(i+1)] = True

        # And load it all in.
        param_priors[superdict_key] = superdict_prior
        fit_or_not[superdict_key] = superdict_fitornot
    
    return param_priors, fit_or_not

def build_bounds(params_priors, priors_type):
    """Builds bounds for linear least squares fitting.

    Args:
        params_priors (dict): each entry is a list of two numbers which are
        either lower/upper limit (for uniform priors) or mean/sigma (gaussian).
        For Gaussian, bounds will be set as the 5-sigma limits.
        priors_type (str): options of 'uniform' or 'gaussian'.

    Returns:
        list: the lower and upper bounds on each parameter to fit.
    """
    # Initialize bounds list.
    bounds = []

    # And unpack.
    '''
    for superdict_key in list(params_priors.keys()):
        # The first key defines the parallel spectrum superdict key.
        for class_key in list(params_priors[superdict_key].keys()):
            # The next key defines the class of prior,
            # e.g. planets, flares, systematics, or lds.
            for key in list(params_priors[superdict_key][class_key].keys()):
                # Finally, we get down to the parameters themselves.
                parameter_priors = params_priors[superdict_key][class_key]
                if priors_type == 'uniform':
                    lower, upper = parameter_priors[key]
                    bounds.append((lower, upper))
                elif priors_type == 'gaussian':
                    mean, sigma = parameter_priors[key]
                    bounds.append((mean-(5*sigma),mean+(5*sigma)))
    '''
    for superdict_key in list(params_priors.keys()):
        # Retrieve the parallel spectrum key.
        for key in list(params_priors[superdict_key].keys()):
            # We get down to the parameters themselves.
            parameter_priors = params_priors[superdict_key][key]
            if priors_type == 'uniform':
                lower, upper = parameter_priors
                bounds.append((lower, upper))
            elif priors_type == 'gaussian':
                mean, sigma = parameter_priors
                bounds.append((mean-(5*sigma),mean+(5*sigma)))
    return bounds

def preservation(planets_fit,
                 preserve_timing=False, preserve_depth=False, preserve_orbit=False):
    if preserve_timing:
        # Arbitrarily call the first spectrum's time-related args as absolute.
        reference_planet = planets_fit['1']
        abs_times = {}
        for k, pl_key in enumerate(list(reference_planet.keys())):
            abs_times[pl_key] = {}
            for time_key in ('t_prim{}'.format(k+1),'t_seco{}'.format(k+1)):
                abs_times[pl_key][time_key] = reference_planet[pl_key][time_key]
        # Now we have taken the t_prim/t_seco args for planet1, planet2, etc. from
        # the first parallelised spectrum as the absolute. All planets must
        # now be fixed to that value.
        for key in list(planets_fit.keys()):
            for k, pl_key in enumerate(list(reference_planet.keys())):
                for time_key in ('t_prim{}'.format(k+1),'t_seco{}'.format(k+1)):
                    # Get spectrum 'key', planet 'pl_key', parameter 'time_key' and lock it.
                    planets_fit[key][pl_key][time_key] = abs_times[pl_key][time_key]
    if preserve_depth:
        # Arbitrarily call the first spectrum's depth-related args as absolute.
        reference_planet = planets_fit['1']
        abs_depths = {}
        for k, pl_key in enumerate(list(reference_planet.keys())):
            abs_depths[pl_key] = {}
            for depth_key in ('rp{}'.format(k+1),'fp{}'.format(k+1)):
                abs_depths[pl_key][depth_key] = reference_planet[pl_key][depth_key]
        # Now we have taken the rp/fp args for planet1, planet2, etc. from
        # the first parallelised spectrum as the absolute. All planets must
        # now be fixed to that value.
        for key in list(planets_fit.keys()):
            for k, pl_key in enumerate(list(reference_planet.keys())):
                for depth_key in ('rp{}'.format(k+1),'fp{}'.format(k+1)):
                    # Get spectrum 'key', planet 'pl_key', parameter 'depth_key' and lock it.
                    planets_fit[key][pl_key][depth_key] = abs_depths[pl_key][depth_key]
    if preserve_orbit:
        # Arbitrarily call the first spectrum's orbit-related args as absolute.
        reference_planet = planets_fit['1']
        abs_orbit = {}
        for k, pl_key in enumerate(list(reference_planet.keys())):
            abs_orbit[pl_key] = {}
            for orbit_key in ('aor{}'.format(k+1),'period{}'.format(k+1),'incl{}'.format(k+1),
                              'ecc{}'.format(k+1),'longitude{}'.format(k+1)):
                abs_orbit[pl_key][orbit_key] = reference_planet[pl_key][orbit_key]
        # Now we have taken the orbit args for planet1, planet2, etc. from
        # the first parallelised spectrum as the absolute. All planets must
        # now be fixed to that value.
        for key in list(planets_fit.keys()):
            for k, pl_key in enumerate(list(reference_planet.keys())):
                for orbit_key in ('aor{}'.format(k+1),'period{}'.format(k+1),'incl{}'.format(k+1),
                                  'ecc{}'.format(k+1),'longitude{}'.format(k+1)):
                    # Get spectrum 'key', planet 'pl_key', parameter 'orbit_key' and lock it.
                    planets_fit[key][pl_key][orbit_key] = abs_orbit[pl_key][orbit_key]
        
    return planets_fit

def log_likelihood(params_array, bundled_params, fit_or_not, exp_times, light_curve, errors,
                   preserve_timing=False, preserve_depth=False, preserve_orbit=False):
    """For emcee. Generates log-likelihood of tested model based on residuals.

    Args:
        params_array (np.array): the input to emcee and what is being fitted.
        bundled_params (dict): all of the original parameters from planets,
        flares, systematics, and lds, spewed out into a long dict.
        fit_or_not (dict): a series of keys explaining which items must be modified.
        exp_times (np.array): timestamps of the mid-exposure times for each point.
        light_curve (np.array): flux at each point in time.
        errors (np.array): uncertainties on the flux to weight the residuals.
        preserve_timing (bool): whether or not to force all time-related arguments
        being fitted to be equal. Useful for parallel fits of simul-events.
        preserve_depth (bool): whether or not to force all depth-related arguments
        being fitted to be equal. Useful for parallel fits of multiple visits.
        preserve_orbit (bool): whether or not to force all orbit-related arguments
        being fitted to be equal. Useful for parallel fits in nearly all cases.

    Returns:
        float: the log-likelihood, metric of how well the model fit the data
        given the uncertainties.
    """
    # This is as simple as calling the residuals.
    residuals = _residuals(params_array, exp_times, light_curve, errors, bundled_params, fit_or_not,
                           preserve_timing, preserve_depth, preserve_orbit, give_res=False)

    # And then multiplying.
    log_l = -0.5*residuals
    return log_l

def log_prior(params_array, priors, priors_type):
    """For emcee. Generates log-prior of tested model based on priors.

    Args:
        params_array (np.array): the input to emcee and what is being fitted.
        priors (np.array): priors on each parameter being fitted.
        priors_type (str): options are "uniform" or "gaussian". Determines
        how log-prior is calculated. For uniform you can get 0 or np.inf,
        while Gaussian priors allow a continuous range of values.

    Returns:
        float: the log-prior, metric of how much we believe the model parameters
        could take these values based on a priori knowledge.
    """
    # Define the prior function.
    def prior_func(parameter, prior, priors_type):
        if priors_type == "gaussian":
            return np.log(1.0/(np.sqrt(2*np.pi)*prior[1]))-0.5*(parameter-prior[0])**2/prior[1]**2
        elif priors_type == "uniform":
            if (prior[0] < parameter and parameter < prior[1]):
                return 0
            else:
                return -np.inf

    # For each parameter, check where it falls in the posterior
    log_p = 0
    for param, prior in zip(params_array, priors):
        log_p += prior_func(param, prior, priors_type)
    
    # Check outcome.
    if np.isnan(log_p):
        # If we got NaN somehow, we don't want these parameters.
        return -np.inf
    elif np.isfinite(log_p):
        # If it is a finite number, then we will take it.
        return log_p
    else:
        # Then it is a nonfinite number and we also don't want it.
        return -np.inf

def log_probability(params_array, bundled_params, fit_or_not, exp_times, light_curve, errors,
                    priors, priors_type, preserve_timing, preserve_depth, preserve_orbit):
    """For emcee. Generates the log-probability, sum of the log-likelihood
    and log-prior.

    Args:
        params_array (np.array): the input to emcee and what is being fitted.
        bundled_params (dict): all of the original parameters from planets,
        flares, systematics, and lds, spewed out into a long dict.
        fit_or_not (dict): a series of keys explaining which items must be modified.
        exp_times (np.array): timestamps of the mid-exposure times for each point.
        light_curve (np.array): flux at each point in time.
        errors (np.array): uncertainties on the flux to weight the residuals.
        priors (np.array): priors on each parameter being fitted.
        priors_type (str): options are "uniform" or "gaussian". Determines
        how log-prior is calculated. For uniform you can get 0 or np.inf,
        while Gaussian priors allow a continuous range of values.
        preserve_timing (bool): whether or not to force all time-related arguments
        being fitted to be equal. Useful for parallel fits of simul-events.
        preserve_depth (bool): whether or not to force all depth-related arguments
        being fitted to be equal. Useful for parallel fits of multiple visits.
        preserve_orbit (bool): whether or not to force all orbit-related arguments
        being fitted to be equal. Useful for parallel fits in nearly all cases.

    Returns:
        float: the log-probability, metric of how likely emcee is to accept
        the move.
    """
    # The log-prior gives us a quick way to decide if residuals are worth checking.
    log_p = log_prior(params_array, priors, priors_type)

    # If the log-prior is not a finite number, these parameters are bad.
    if np.isnan(log_p):
        return -np.inf
    if not np.isfinite(log_p):
        return -np.inf
    
    # Let's go get the log-likelihood then.
    return log_p + log_likelihood(params_array, bundled_params, fit_or_not, exp_times, light_curve, errors,
                                  preserve_timing, preserve_depth, preserve_orbit)

def get_result_from_post(ndim, flat_samples):
    params_array = []
    param_errs_array = []
    for i in range(ndim):
        params_array.append(np.percentile(flat_samples[:, i], 50))
        param_errs_array.append(np.std(flat_samples[:, i]))
    return np.array(params_array), np.array(param_errs_array)

def _residuals(params_array, exp_times, light_curve, errors, bundled_params, fit_or_not,
               preserve_timing=False, preserve_depth=False, preserve_orbit=False,
               give_res=False):
    """Computes the residuals between the full model and the given light curve.

    Args:
        params_array (np.array): the array-ified version of params_to_fit.
        exp_times (np.array): mid-exposure time of each point in the light curve.
        light_curve (np.array): flux at each point in the light curve.
        errors (np.array): uncertainties associated with each data point, used
        in weighting the residuals.
        bundled_params (dict): all of the original parameters from planets,
        flares, systematics, and lds, spewed out into a long dict.
        fit_or_not (dict): a series of keys explaining which items must be modified.
        preserve_timing (bool): whether or not to force all time-related arguments
        being fitted to be equal. Useful for parallel fits of simul-events.
        preserve_depth (bool): whether or not to force all depth-related arguments
        being fitted to be equal. Useful for parallel fits of multiple visits.
        preserve_orbit (bool): whether or not to force all orbit-related arguments
        being fitted to be equal. Useful for parallel fits in nearly all cases.
        give_res (bool, optional): if asked, return the residuals as an array, not summed.
        Defaults to False.
    
    Returns:
        float or np.array: if not give_res, returns the summed residuals to
        evaluate the goodness of fit. If give_res, returns the residuals array.
    """
    # Turn the array back into a dictionary.
    redicted_params = array_to_dict(params_array, bundled_params, fit_or_not)
    
    # Then, separate those back into planets, flares, and systematics.
    planets_fit, flares_fit, systematics_fit, ld_fit = {}, {}, {}, {}
    for key in list(redicted_params.keys()):
        planets_fit[key], flares_fit[key], systematics_fit[key], ld_fit[key] = unpack_params_back_to_dicts(redicted_params[key])

    # Force equals where called for using the preserve arguments.
    planets_fit = preservation(planets_fit,
                               preserve_timing, preserve_depth, preserve_orbit)

    # Now redo the flux model calculation for each spectrum,
    # this time supplying redicted_params as an argument.
    sum_residuals = 0
    full_residuals = []
    for d, superdict_key in enumerate(list(planets_fit.keys())):
        model, components = models.full_model(exp_times[d],
                                              planets_fit[superdict_key],
                                              flares_fit[superdict_key],
                                              systematics_fit[superdict_key],
                                              bundled_params=redicted_params[superdict_key],
                                              fit_or_not=fit_or_not[superdict_key])
        # And compare to the data.
        residuals_full = (model-light_curve[d])/errors[d]
        full_residuals.append(residuals_full)
        sum_residuals += np.sum(residuals_full**2)
    
    if give_res:
        return full_residuals
    else:
        return sum_residuals
