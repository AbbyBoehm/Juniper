import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erfc

from juniper.util.cleaning import median_timeseries_filter
from juniper.stage5 import batman_handler

def full_model(t, planets, flares, systematics, bundled_params=None, fit_or_not=None):
    """Builds a model of a transit/eclipse light curve using the provided
    information on observed planets, suspected flaring events, and systematic
    models to detrend for.

    Args:
        t (np.array): time.
        planets (dict): a series of dictionary entries describing each planet
        in the transit curve. They have been batman-initialized so they already
        contain parameters "batman_model" and "batman_params".
        flares (dict): a series of dictionary entries describing each flaring
        event suspected to have occurred during the observation.
        systematics (dict): a series of dictionary entries describing each
        systematic model to detrend for.
        bundled_params (dict): all of the original parameters from planets,
        flares, systematics, and lds, spewed out into a long dict.
        fit_or_not (dict): a series of keys explaining which items must be modified.
    
    Returns:
        np.array, dict: Sys(t;A)*(Sum[planets(t;B)+flares(t;C)]), or the sum of
        planet and flare events multiplied by systematic detrending. Every component
        of the model is also output individually as a dictionary.
    """
    # Initialize planett flux against time as 1s.
    flx = np.ones_like(t)
    # Track individual models as well.
    models = {}

    # Build planetary flux.
    for planet_name in list(planets.keys()):
        planet = planets[planet_name] # dict, contains rp, rp_prior, fp, fp_prior, etc. as well as batman_model
        planet_ID = str.replace(planet_name, "planet", "")
        batman_flux = batman_handler.batman_flux_update(bundled_params=bundled_params,
                                                        fit_or_not=fit_or_not,
                                                        batman_params=[planet["batman_params"+planet_ID],],
                                                        batman_model=[planet["batman_model"+planet_ID],])
        # Multiply planet's flux contribution into the full model.
        flx *= batman_flux
        models[planet_name] = batman_flux
    
    # Build flare flux.
    for flare_ID in list(flares.keys()):
        flare = flares[flare_ID] # dict, contains flare parameters
        flare_flux = flare_model(t, flare, str.replace(flare_ID,"flare",""))
        # Multiply flare's flux contribution into the full model.
        flx *= flare_flux
        models[flare_ID] = flare_flux

    # Build systematics modifier.
    system = np.ones_like(t)

    # Polynomial trend.
    if systematics["poly"]:
        poly = systematic_polynomial(t, systematics["poly_coeffs"])
        system *= poly
        models["poly"] = poly

    # Mirror tilt event.
    if systematics["mirrortilt"]:
        mirrortilt = systematic_mirrortilt(t, systematics["mirrortilt_coeffs"])
        system *= mirrortilt
        models["mirrortilt"] = mirrortilt

    # Single ramp fit.
    if systematics["singleramp"]:
        expramp = systematic_expramp(t, systematics["singleramp_coeffs"])
        system *= expramp
        models["singleramp"] = expramp

    # Double ramp fit.
    if systematics["doubleramp"]:
        expramp = systematic_doubleramp(t, systematics["doubleramp_coeffs"])
        system *= expramp
        models["doubleramp"] = expramp

    # Dispersion position detrend.
    if systematics["disp_detrend"]:
        jitter_x = systematic_jitter_disp(systematics["xpos"],
                                          systematics["disp_detrend_coeffs"])
        # If building an interpolated model, these ones can have size mismatch.
        if len(jitter_x) != len(system):
            jitter_x = interpolate_model(jitter_x, system)
        system *= jitter_x
        models["disp_detrend"] = jitter_x

    # Cross-dispersion position detrend.
    if systematics["spatial_detrend"]:
        jitter_y = systematic_jitter_crossdisp(systematics["ypos"],
                                               systematics["spatial_detrend_coeffs"])
        # If building an interpolated model, these ones can have size mismatch.
        if len(jitter_y) != len(system):
            jitter_y = interpolate_model(jitter_y, system)
        system *= jitter_y
        models["spatial_detrend"] = jitter_y

    # Width detrend.
    if systematics["width_detrend"]:
        psf = systematic_psf(systematics["width"],
                             systematics["width_detrend_coeffs"])
        # If building an interpolated model, these ones can have size mismatch.
        if len(psf) != len(system):
            psf = interpolate_model(psf, system)
        system *= psf
        models["width_detrend"] = psf

    # And fold all together.
    return system*flx, models

def systematic_polynomial(t, coeffs):
    """Returns a polynomial of specified order in time.

    Args:
        t (np.array): time.
        coeffs (list): polynomial coefficients.

    Returns:
        np.array: polynomial model to be added to Sys(t;A).
    """
    # Set up empty polynomial.
    poly = np.array([0 for i in t], dtype='float64')

    # And populate.
    for n, o in enumerate(coeffs):
        poly += np.array(o*((t-t[0])**n), dtype='float64')
    
    return poly

def systematic_expramp(t, coeffs):
    """Returns a single exponential ramp trend in time.

    Args:
        t (np.array): time.
        coeffs (list): exponential ramp coefficients.

    Returns:
        np.array: single ramp model to be added to Sys(t;A).
    """
    single_ramp = 1 + coeffs[0]*np.exp(coeffs[1]*(t-t[0]) + coeffs[2])
    return single_ramp

def systematic_doubleramp(t, coeffs):
    """Returns a double exponential ramp trend in time.

    Args:
        t (np.array): time.
        coeffs (list): exponential ramp coefficients.

    Returns:
        np.array: double ramp model to be added to Sys(t;A).
    """
    double_ramp = (1 + coeffs[0]*np.exp(coeffs[1]*(t-t[0]) + coeffs[2])
                     + coeffs[3]*np.exp(-coeffs[4]*(t-t[0]) + coeffs[5]))
    return double_ramp

def systematic_mirrortilt(t, coeffs):
    """Returns a step function modelling an arbitrary number
     of mirror tilt events.

    Args:
        t (np.array): time.
        coeffs (list): mirror tilt coefficients.
    
    Returns:
        np.array: mirror tilts model to be added to Sys(t;A).
    """
    flx = coeffs[0]*np.ones_like(t) # there is a pre-tilt baseline flux [0]
    for n in range(1,len(coeffs)):
        flx[coeffs[n][0]:] += coeffs[n][1] # and then after time index [n][0], there is a step of [n][1] which can be up or down
    return flx

def systematic_jitter_disp(xpos, coeffs):
    """Returns a polynomial correlated to trace x position.

    Args:
        xpos (np.array): dispersion position with time.
        coeffs (list): x-jitter polynomial fits.
    
    Returns:
        np.array: x-jitter model to be added to Sys(t;A).
    """
    jitter = 1 + coeffs[0]*xpos
    return jitter

def systematic_jitter_crossdisp(ypos, coeffs):
    """Returns a polynomial correlated to trace y position.

    Args:
        ypos (np.array): cross-dispersion position with time.
        coeffs (list): y-jitter polynomial fits.
    
    Returns:
        np.array: y-jitter model to be added to Sys(t;A).
    """
    jitter = 1 + coeffs[0]*ypos
    return jitter

def systematic_psf(widths, coeffs):
    """Returns a polynomial correlated to trace width.

    Args:
        widths (np.array): cross-dispersion width with time.
        coeffs (list): width polynomial fits.
    
    Returns:
        np.array: psf model to be added to Sys(t;A).
    """
    psf = 1 + coeffs[0]*widths
    return psf

def flare_model(t, flare, flare_ID):
    """Model of a flare from Tovar Mendoza+ 2022

    Args:
        t (np.array): time.
        flare (dict): description of this flare, including its start time,
        amplitude, and fade time.
        flare_ID (str): the number of the flare. Used for picking the right keys.

    Returns:
        np.array: flux of a flare with time.
    """
    t_offset = np.array([ti - flare["E"+flare_ID] for ti in t])
    c1 = np.sqrt(np.pi)*flare["A"+flare_ID]*flare["C"+flare_ID]/2
    c2 = flare["Fr"+flare_ID]*flare_h(t_offset, flare["B"+flare_ID], flare["C"+flare_ID], flare["Dr"+flare_ID])
    c3 = (1-flare["Fr"+flare_ID])*flare_h(t_offset, flare["B"+flare_ID], flare["C"+flare_ID], flare["Ds"+flare_ID])
    return c1*(c2+c3)

def flare_h(t, B, C, D):
    """The exponetial h terms from Tovar Mendoza+ 2022's flare model.

    Args:
        t (np.array): time.
        B (float): parameter of the h term.
        C (float): parameter of the h term.
        D (float): parameter of the h term.

    Returns:
        np.array: an h term in the flare.
    """
    a1 = -D*t
    a2 = D*C/2
    a3 = ((B/C)+a2)**2
    a4 = (B-t)/C
    return np.exp(a1+a3)*erfc(a4+a2)

def interpolate_model(model, base):
    """Interpolates the model to match the time resolution of the base.

    Args:
        model (np.array): model that may be a position or width detrend.
        base (np.array): interpolated system model.

    Returns:
        np.array: model interpolated to match the base resolution.
    """
    # Define an interpolater function.
    x = np.linspace(0,len(base),len(model))
    xnew = np.linspace(0,len(base),len(base))
    interp_model = interp1d(x, model, kind='linear', fill_value='extrapolate')
    return interp_model(xnew)