import batman

def batman_transit_params(exoplanet_params, planet_ID, ld_coeffs, model_type):
    """Simple function to set up the batman.TransitParams() object.

    Args:
        exoplanet_params (dict): exoplanet parameters.
        planet_ID (str): the ID of this planet.
        ld_coeffs (list): the limb darkening coefficients.
        model_type (str): the type of limb darkening model in use.

    Returns:
        batman.TransitParams(): parameters for batman model.
    """
    params = batman.TransitParams()
    params.per = exoplanet_params["period"+planet_ID]                                 #orbital period in days
    params.rp = exoplanet_params["rp"+planet_ID]                                      #planet radius (in units of stellar radii)
    params.t0 = exoplanet_params["t_prim"+planet_ID]                                  #time of inferior conjunction in days
    params.a = exoplanet_params["aor"+planet_ID]                                      #semi-major axis (in units of stellar radii)
    params.inc = exoplanet_params["incl"+planet_ID]                                   #orbital inclination (in degrees)
    params.ecc = exoplanet_params["ecc"+planet_ID]                                    #eccentricity
    params.w = exoplanet_params["longitude"+planet_ID]                                #longitude of periastron (in degrees)
    params.u = ld_coeffs                                                              #limb darkening coefficients [u1, u2] or [u1, u2, u3, u4] etc.
    params.limb_dark = model_type                                                     #limb darkening model

    return params

def batman_eclipse_params(exoplanet_params, planet_ID):
    """Simple function to set up the batman.TransitParams() object.

    Args:
        exoplanet_params (dict): exoplanet parameters.
        planet_ID (str): the ID of this planet.

    Returns:
        batman.TransitParams(): parameters for batman model.
    """
    params = batman.TransitParams()
    params.per = exoplanet_params["period"+planet_ID]                                 #orbital period in days
    params.rp = exoplanet_params["rp"+planet_ID]                                      #planet radius (in units of stellar radii)
    params.fp = exoplanet_params["fp"+planet_ID]                                      #planet flux (in units of stellar flux)
    params.t_secondary = exoplanet_params["t_seco"+planet_ID]                         #time of superior conjunction in days
    params.a = exoplanet_params["aor"+planet_ID]                                      #semi-major axis (in units of stellar radii)
    params.inc = exoplanet_params["incl"+planet_ID]                                   #orbital inclination (in degrees)
    params.ecc = exoplanet_params["ecc"+planet_ID]                                    #eccentricity
    params.w = exoplanet_params["longitude"+planet_ID]                                #longitude of periastron (in degrees)
    params.limb_dark = "uniform"                                                      #for batman eclipse modelling, you still have to give it lds even if it isn't using them
    params.u = []                                                                     #for batman eclipse modelling, you still have to give it lds even if it isn't using them

    return params

def batman_init_one_model(t, exoplanet_params, event, planet_ID, ld_initialguess, model_type):
    """Simple function to initialize a batman transit or eclipse model.

    Args:
        t (np.array): time.
        exoplanet_params (dict): exoplanet parameters which tells
        batman how to build the model.
        event (str): options are 'primary' or 'secondary'.
        planet_ID (str): the ID of this planet.
        ld_initialguess (list): the limb darkening coefficients.
        model_type (str): the type of limb darkening model in use.

    Returns:
        batman.TransitModel(): batman transit model for the transit or eclipse.
    """
    # Translate exoplanet parameters to batman.TransitParams() object.
    if event == 'primary':
        batman_params = batman_transit_params(exoplanet_params, planet_ID, ld_initialguess, model_type)
    if event == 'secondary':
        batman_params = batman_eclipse_params(exoplanet_params, planet_ID)

    # Initialize a batman_model.
    batman_model = batman.TransitModel(batman_params, t, transittype=event)
    return batman_model, batman_params

def batman_init_all_planets(t, planets, ld, event):
    """Wrapper to init models for all planets.

    Args:
        t (np.array): time.
        planets (dict): a series of dictionary entries describing each planet
        in the transit or eclipse curve.
        ld (dict): instructions on handling stellar limb darkening, necessary for
        talking to batman.
        event (str): options are 'primary' or 'secondary'.

    Returns:
        dict: planets updated with keywords "batman_model" and "batman_params".
    """
    # Grab the ld info we need to properly initialize batman.
    ld_coeffs = ld["ld_coeffs"]
    model_type = ld["ld_model"]

    # Update planets to have batman models and parameters.
    for planet_name in list(planets.keys()):
        # Grab the planet-specific dict.
        planet = planets[planet_name]

        # Supply its ID number so that we can read out the right tags.
        planet_ID = str.replace(planet_name,"planet","")
        planet["batman_model"+planet_ID], planet["batman_params"+planet_ID] = batman_init_one_model(t, planet, event, planet_ID, ld_coeffs, model_type)
    return planets

# You ever stare at a screen so long you stop noticing the word 'batman'?

def batman_flux_update(bundled_params, fit_or_not, batman_params, batman_model):
    """Simple function to get the new batman flux model.

    Args:
        bundled_params (dict): all of the original parameters from planets,
        flares, systematics, and lds, spewed out into a long dict. Can also be
        supplied as None when just retrieving the batman flux.
        fit_or_not (dict): a series of keys explaining which items must be modified.
        batman_params (list): list of batman.TransitParams() objects to update
        and supply to the batman_model objects.
        batman_model (list): batman.TranstiModel() objects which return
        transit/eclipse flux when supplied with parameters.

    Returns:
        np.array: total flux for the transit/eclipse events.
    """
    # Update the batman_params if asked.
    if bundled_params:
        # Need to update the params for each model.
        for i, (batman_params_i, batman_model_i) in enumerate(zip(batman_params,batman_model)):
            batman_params_i = update_batman_params(bundled_params, fit_or_not, batman_params_i, str(i+1))
    
    # And calculate and sum bat_flux.
    for i, (batman_params_i, batman_model_i) in enumerate(zip(batman_params,batman_model)):
        if i == 0:
            bat_flux = batman_model_i.light_curve(batman_params_i)
        else:
            bat_flux += batman_model_i.light_curve(batman_params_i)
    return bat_flux

def update_batman_params(bundled_params, fit_or_not, batman_params, planet_ID):
    """Simple function to help the bundled_params dictionary talk
    to the batman.TransitParams() object.

    Args:
        bundled_params (dict): all of the original parameters from planets,
        flares, systematics, and lds, spewed out into a long dict.
        fit_or_not (dict): a series of keys explaining which items must be modified.
        batman_params (list): list of batman.TransitParams() objects to update
        and supply to the batman_model objects.
        batman_params (batman.TransitParams()): batman.TransitParams() object
        which needs to be updated
        planet_ID (str): number of the planet being worked on. Helps grab
        the correct tags.
        
    Returns:
        batman.TransitParams(): updated parameters for batman.
    """
    # Check through fit_or_not and, where told to do so,
    # use the relevant bundled_params info to update batman.
    if fit_or_not["rp_prior"+planet_ID]:
        batman_params.rp = bundled_params["rp"+planet_ID]
    if fit_or_not["fp_prior"+planet_ID]:
        batman_params.fp = bundled_params["fp"+planet_ID]
    if fit_or_not["t_prim_prior"+planet_ID]:
        batman_params.t0 = bundled_params["t_prim"+planet_ID]
    if fit_or_not["t_seco_prior"+planet_ID]:
        batman_params.t_secondary = bundled_params["t_seco"+planet_ID]
    if fit_or_not["period_prior"+planet_ID]:
        batman_params.per = bundled_params["period"+planet_ID]
    if fit_or_not["aor_prior"+planet_ID]:
        batman_params.a = bundled_params["aor"+planet_ID]
    if fit_or_not["incl_prior"+planet_ID]:
        batman_params.inc = bundled_params["incl"+planet_ID]
    if fit_or_not["ecc_prior"+planet_ID]:
        batman_params.ecc = bundled_params["ecc"+planet_ID]
    if fit_or_not["longitude_prior"+planet_ID]:
        batman_params.w = bundled_params["longitude"+planet_ID]

    # ld has a slight bit of nuance to it.
    if any(bundled_params["fit_lds"]):
        # This will have been updated before.
        batman_params.u = bundled_params["ld_coeffs"]
    
    return batman_params