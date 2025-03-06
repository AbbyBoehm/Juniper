import numpy as np

def compute_depth_rprs(fitted_dict, fitted_errs_dict, planet_ID):
    """Retrieve the simple Rp/Rs depth.

    Args:
        fitted_dict (dict): dictionary of fitted parameters.
        fitted_errs_dict (dict): dictionary of uncertainties on fitted parameters.
        planet_ID (str): the ID of planet we are asking for.

    Returns:
        tuple: the depth Rp/Rs and its uncertainty.
    """
    depth = fitted_dict["rp"+planet_ID]
    err = fitted_errs_dict["rp"+planet_ID]
    return depth, err

def compute_depth_rprs2(fitted_dict, fitted_errs_dict, planet_ID):
    """Retrieve the very slightly complicated (Rp/Rs)**2 depth and uncertainty.

    Args:
        fitted_dict (dict): dictionary of fitted parameters.
        fitted_errs_dict (dict): dictionary of uncertainties on fitted parameters.
        planet_ID (str): the ID of planet we are asking for.

    Returns:
        tuple: the depth (Rp/Rs)^2 and its uncertainty.
    """
    depth = fitted_dict["rp"+planet_ID]**2
    err = 2*fitted_dict["rp"+planet_ID]*fitted_errs_dict["rp"+planet_ID]
    return depth, err

def compute_depth_aoverlap(fitted_dict, fitted_errs_dict, planet_ID):
    """Retrieve the rather complicated planet-star overlap depth and uncertainty.

    Args:
        fitted_dict (dict): dictionary of fitted parameters.
        fitted_errs_dict (dict): dictionary of uncertainties on fitted parameters.
        planet_ID (str): the ID of planet we are asking for.

    Returns:
        tuple: the depth A-overlap and its uncertainty.
    """
    # First, get the inclination.
    inc = fitted_dict["incl"+planet_ID]*np.pi/180
    inc_err = fitted_errs_dict["incl"+planet_ID]*np.pi/180
    if inc == inc_err:
        # Catch whether this was fit at all.
        inc_err = 0
    
    # Then the a/R*.
    aoR = fitted_dict["aor"+planet_ID]
    aoR_err = fitted_errs_dict["aor"+planet_ID]
    if aoR == aoR_err:
        # Catch whether this was fit at all.
        aoR_err = 0
    
    # Compute the b parameter.
    bo = aoR*np.cos(inc)
    bo_sq = bo**2
    bo_err = np.sqrt((np.cos(inc)*aoR_err)**2 + (aoR*np.sin(inc)*inc_err)**2)
    
    # Stellar radius is just 1 in this system.
    rs = 1
    rs_sq = rs**2

    # Planet radius is in units of stellar radius.
    rp = fitted_dict["rp"+planet_ID]
    rp_sq = rp**2
    rp_err = fitted_errs_dict["rp"+planet_ID]

    # Compute phi arguments.
    arg_phi_1 = ((bo_sq * rs_sq) + rp_sq - rs_sq)/(2 * (bo * rs) * rp)
    arg_phi_1_err = np.sqrt((bo_err * (bo_sq * rs_sq - rp_sq + rs_sq) / (2 * bo_sq * rp * rs))**2 + 
                            (rp_err * (-bo_sq * rs_sq + rp_sq + rs_sq) / (2 * bo * rp_sq * rs))**2)

    arg_phi_2 = ((bo_sq * rs_sq) + rs_sq - rp_sq)/(2 * (bo * rs) * rs)
    arg_phi_2_err = np.sqrt((bo_err * (bo_sq * rs_sq + rp_sq - rs_sq) / (2 * bo_sq * rs_sq))**2 + 
                            (rp_err * rp / (bo * rs_sq))**2)

    # And supply them to arc-cosine.
    phi_1 = np.arccos(arg_phi_1)  # Angle at planet centre
    phi_1_err = np.sqrt(arg_phi_1_err**2 / (1 - arg_phi_1))

    phi_2 = np.arccos(arg_phi_2)  # Angle at star centre
    phi_2_err = np.sqrt(arg_phi_2_err**2 / (1 - arg_phi_2))

    # Evaluate the overlapping area analytically.
    A_overlap = (rp_sq * (phi_1 - 0.5 * np.sin(2.0 * phi_1)) +
                    rs_sq * (phi_2 - 0.5 * np.sin(2.0 * phi_2)))
    A_s = np.pi*rs_sq
    depth = A_overlap/A_s

    # The tedious process of computing the depth error...
    dAdrp = 2*rp*(phi_1 - 0.5 * np.sin(2.0 * phi_1))/A_s
    dAdphi_1 = rp_sq*(1 - np.cos(2.0 * phi_1))/A_s
    dAdphi_2 = rs_sq*(1 - np.cos(2.0 * phi_2))/A_s
    err = np.sqrt((dAdrp*rp_err)**2 + (dAdphi_1*phi_1_err)**2+(dAdphi_2*phi_2_err)**2)

    # And return!
    return depth, err

def compute_depth_fpfs(fitted_dict, fitted_errs_dict, planet_ID):
    """Retrieve the simple Fp/Fs depth.

    Args:
        fitted_dict (dict): dictionary of fitted parameters.
        fitted_errs_dict (dict): dictionary of uncertainties on fitted parameters.
        planet_ID (str): the ID of planet we are asking for.

    Returns:
        tuple: the depth Fp/Fs and its uncertainty.
    """
    depth = fitted_dict["fp"+planet_ID]
    err = fitted_errs_dict["fp"+planet_ID]
    return depth, err