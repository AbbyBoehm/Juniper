import numpy as np

from exotic_ld import StellarLimbDarkening as SLD

def get_exotic_coefficients(exoticLD_instructions):
    """Get limb darkening coefficients from ExoTiC-LD.

    Args:
        exoticLD_instructions (dict): instructions for limb darkening handling.

    Returns:
        list: list of floats corresponding to calculated ld coefficients.
    """
    # Unpack instructions.
    model_type = exoticLD_instructions["ld_model"]
    stellar_params = exoticLD_instructions["stellar_params"]
    wavelength_range = 1e4*exoticLD_instructions["wavelength_range"] # factor 1e4 converts micron to AA
    instrument_mode = exoticLD_instructions["instrument_mode"]
    ld_data_path = exoticLD_instructions["ld_data_path"]
    ld_grid = exoticLD_instructions["ld_grid"]
    custom_grid = exoticLD_instructions["custom_grid"]
    interpolate = exoticLD_instructions["interpolate"]

    print("Generating custom lds for wavelength range [AA]:", wavelength_range)

    # Check for custom model, indicated by ld_grid == None. If there is a custom model,
    # generate SLD from that.
    if not ld_grid:
        s_wvs = (np.genfromtxt(custom_grid, skip_header = 2, usecols = [0]).T)*1e4
        s_mus = np.flip(np.genfromtxt(custom_grid, skip_header = 1, max_rows = 1))
        stellar_intensity = np.flip(np.genfromtxt(custom_grid, skip_header = 2)[:,1:],axis = 1)

        sld = SLD(ld_data_path=ld_data_path, ld_model="custom",
                  custom_wavelengths=s_wvs, custom_mus=s_mus, custom_stellar_model=stellar_intensity)
    # Otherwise, generate SLD from the stellar params.
    else:
        sld = SLD(M_H=stellar_params["MH"], Teff=stellar_params["Teff"], logg=stellar_params["logg"],
                  ld_model=ld_grid, ld_data_path=ld_data_path, interpolate_type=interpolate, verbose=True)
    
    # Now use SLD to make coefficients for the requested model type.
    if model_type == "linear":
        lds = sld.compute_linear_ld_coeffs(wavelength_range=wavelength_range,
                                           mode=instrument_mode,mu_min=0.0)
    if model_type == "quadratic":
        lds = sld.compute_quadratic_ld_coeffs(wavelength_range=wavelength_range,
                                              mode=instrument_mode,mu_min=0.0)
    if model_type == "square-root":
        lds = sld.compute_squareroot_ld_coeffs(wavelength_range=wavelength_range,
                                               mode=instrument_mode,mu_min=0.0)
    if model_type == "nonlinear":
        lds = sld.compute_4_parameter_non_linear_ld_coeffs(wavelength_range=wavelength_range,
                                                           mode=instrument_mode,mu_min=0.0)
    lds = [i for i in lds]
    return lds