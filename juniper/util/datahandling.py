import os
from tqdm import tqdm

import numpy as np
from astropy.io import fits

from jwst import datamodels as dm

def stitch_files(files, time_step, verbose):
    """Reads all supplied *calints.fits files and stitches them together into a dictionary.

    Args:
        files (lst of str): filepaths to files that are to be loaded.
        time_step (bool): whether to report timing with tqdm.
        verbose (int): from 0 to 2. How much logging to do.

    Returns:
        dict: loaded data.
    """
    # Log.
    if verbose >= 1:
        print("Stitching data *calints.fits files together for Stage 3 processing...")
    
    if verbose == 2:
        print("Will stitch together the following files:")
        for i, f in enumerate(files):
            print(i, f)

    # Initialize some empty lists.
    data, err, jwstdq, wav = [], [], [], [], # the data
    time = [] # the coords
    int_count, insts, dets, filters, gratings = [], [], [], [], [] # the attributes

    # Read in each file.
    for file in tqdm(files,
                     desc = 'Stitching files...',
                     disable=(not time_step)):
        data_i, err_i, int_count_i, wav_i, jwstdq_i, t_i,\
            obs_instrument_i, obs_detector_i, obs_filter_i, obs_grating_i = read_one_datamodel(file)
        wav_i = [wav_i for i in range(t_i.shape[0])] # wavelength solution does not change,
                                                     # but must be extended across all time
        if obs_grating_i == "MIR_LRS-SLITLESS":
            # It's MIRI. Rotate it.
            if verbose == 2:
                print("MIRI LRS file found. Rotating arrays...")
            previous_shape = np.shape(data_i)
            data_i = np.rot90(data_i,k=3,axes=(1,2))
            new_shape = np.shape(data_i)

            err_i = np.rot90(err_i,k=3,axes=(1,2))
            wav_i = np.rot90(wav_i,k=3,axes=(1,2))
            jwstdq_i = np.rot90(jwstdq_i,k=3,axes=(1,2))
            print("Shape changed from {} to {}.".format(previous_shape,new_shape))
        
        # These do not need to be unpacked and can be appended as is.
        int_count.append(int_count_i)
        insts.append(obs_instrument_i)
        dets.append(obs_detector_i)
        filters.append(obs_filter_i)
        gratings.append(obs_grating_i)

        # The rest needs to be unpacked.
        for i in range(data_i.shape[0]):
            data.append(data_i[i])
            err.append(err_i[i])
            jwstdq.append(jwstdq_i[i])
            time.append(t_i[i])
            wav.append(wav_i[i])

    # Now convert to dict.
    segments = {"data":np.array(data),
                "err":np.array(err),
                "badpixmask":np.zeros_like(data),
                "jwstdq":np.array(jwstdq),
                "junidq":np.zeros_like(jwstdq),
                "wavelengths":np.array(wav),
                "time":np.array(time),
                "integrations":int_count,
                "insts":insts,
                "detectors":dets,
                "filters":filters,
                "gratings":gratings,}

    # Log.
    if verbose >= 1:
        print("Files stitched together into dictionary.")
    
    return segments
    
def read_one_datamodel(file):
    """Read one .fits file as a datamodel and return its attributes.

    Args:
        file (str): path to the .fits file you want to read out.

    Returns:
        np.array, np.array, int, np.array, np.array, np.array: the data, errors, integration count, wavelength solution, data quality array, and exposure mid-times.
    """
    with dm.open(file) as f:
         # Get data.
         data = f.data
         err = f.err
         int_count = data.shape[0]
         wav = f.wavelength
         dq = f.dq
         t = f.int_times["int_mid_BJD_TDB"]

         # And get observation details.
    with fits.open(file) as f:
         obs_instrument = f[0].header["INSTRUME"]
         obs_detector = f[0].header["DETECTOR"]
         obs_filter = f[0].header["FILTER"]
         try:
            obs_grating = f[0].header["GRATING"]
         except KeyError:
             obs_grating = f[0].header["EXP_TYPE"]
    
    return data, err, int_count, wav, dq, t, obs_instrument, obs_detector, obs_filter, obs_grating

def stitch_npys(files, time_step, verbose):
    """Reads all supplied *calints.fits files and stitches them together into a dictionary.

    Args:
        files (lst of str): filepaths to files that are to be loaded.
        time_step (bool): whether to report timing with tqdm.
        verbose (int): from 0 to 2. How much logging to do.

    Returns:
        dict: loaded data.
    """
    # Log.
    if verbose >= 1:
        print("Stitching .npy files together for Stage 4 processing...")
    
    if verbose == 2:
        print("Will stitch together the following files:")
        for i, f in enumerate(files):
            print(i, f)

    # Initialize some empty lists.
    data, err, badpixmask, jwstdq, junidq, \
        cdisp, disp, cwidth, wav = [], [], [], [], [], [], [], [], [] # the data
    time = [] # the coords
    int_count, flagged, insts, dets, filters, gratings = [], [], [], [], [], [] # the attributes

    # Read in each file.
    for file in tqdm(files,
                     desc = 'Stitching files...',
                     disable=(not time_step)):
        data_i, err_i, badpixmask_i, jwstdq_i, junidq_i, \
            disp_i, cdisp_i, cwidth_i, wav_i, t_i, \
                flagged_i, int_count_i, obs_instrument_i, obs_detector_i, obs_filter_i, obs_grating_i = read_one_postproc(file)

        # These do not need to be unpacked and can be appended as is.
        int_count.append(int_count_i)
        flagged.append(flagged_i)
        insts.append(obs_instrument_i)
        dets.append(obs_detector_i)
        filters.append(obs_filter_i)
        gratings.append(obs_grating_i)

        # The rest needs to be unpacked.
        for i in range(data_i.shape[0]):
            data.append(data_i[i])
            err.append(err_i[i])
            badpixmask.append(badpixmask_i[i])
            jwstdq.append(jwstdq_i[i])
            junidq.append(junidq_i[i])
            time.append(t_i[i])
            disp.append(disp_i[i])
            cdisp.append(cdisp_i[i])
            cwidth.append(cwidth_i[i])
            wav.append(wav_i[i])

    # Now convert to dict.
    segments = {"data":np.array(data),
                "err":np.array(err),
                "badpixmask":np.array(badpixmask),
                "jwstdq":np.array(jwstdq),
                "junidq":np.array(junidq),
                "disp":np.array(disp),
                "cdisp":np.array(cdisp),
                "cwidth":np.array(cwidth),
                "wavelengths":np.array(wav),
                "time":np.array(time),
                "integrations":int_count,
                "flagged":flagged,
                "insts":insts,
                "detectors":dets,
                "filters":filters,
                "gratings":gratings,}

    # Log.
    if verbose >= 1:
        print("Files stitched together into dictionary.")
    
    return segments


def read_one_postproc(file):
    """Read one post-processing .npy file and return its attributes.

    Args:
        file (str): path to the .npy file you want to read out.

    Returns:
        np.array, np.array, int, np.array, np.array, np.array, np.array, np.array, \
        np.array, np.array: the data, errors, integration count, wavelength solution, \
        data quality array, exposure mid-times, cross/dispersion positions and widths, \
        and frame numbers flagged for motion.
    """
    segment = np.load(file,allow_pickle=True).item()
    data = segment["data"]
    err = segment["err"]
    badpixmask = segment["badpixmask"]
    int_count = segment["integrations"]
    wav = segment["wavelengths"]
    jwstdq = segment["jwstdq"]
    junidq = segment["junidq"]
    time = segment["time"]
    disp = segment["disp"]
    cdisp = segment["cdisp"]
    cwidth = segment["cwidth"]
    flagged = segment["flagged"]
    insts = segment["insts"]
    dets = segment["detectors"]
    filters = segment["filters"]
    gratings = segment["gratings"]
    return data, err, badpixmask, jwstdq, junidq, \
            disp, cdisp, cwidth, wav, time, \
                flagged, int_count, insts, dets, filters, gratings

def save_s3_output(segments, disp_pos, cdisp_pos, cdisp_widths, moved_ints, outfiles, outdir):
    """Saves a .npy for every cleaned segment in the stitched-together files.

    Args:
        segments (dictionary): the dictionary generated by stitch_files.
        disp_pos (list): dispersion positions. Could be an empty list.
        cdisp_pos (list): cross-dispersion positions. Could be an empty list.
        cdisp_widths (list): cross-dispersion widths. Could be an empty list.
        moved_ints (list): integrations flagged for movement. Could be an empty list.
        outfiles (list of str): names for each output file.
        outdir (str): directory to save the output files to.
    """
    # For every segment in the array, we need to break it up.
    int_left = 0
    int_right = 0
    for i, (ints, outfile) in enumerate(zip(segments["integrations"], outfiles)):
        # Get the next limit.
        int_right += ints

        # Snip just the data that we need.
        data = segments["data"][int_left:int_right,:,:]
        err = segments["err"][int_left:int_right,:,:]
        badpixmask = segments["badpixmask"][int_left:int_right,:,:]
        jwstdq = segments["jwstdq"][int_left:int_right,:,:]
        junidq = segments["junidq"][int_left:int_right,:,:]
        time = segments["time"][int_left:int_right]
        wavelengths = segments["wavelengths"][int_left:int_right,:,:]
        insts = segments["insts"][i]
        dets = segments["detectors"][i]
        filters = segments["filters"][i]
        gratings = segments["gratings"][i]

        # Plus the new tracking data, if there is any.
        disp = np.zeros_like(time)
        if disp_pos:
            disp = disp_pos[int_left:int_right]
        cdisp = np.zeros_like(time)
        if cdisp_pos:
            cdisp = cdisp_pos[int_left:int_right]
        cwidth = np.zeros_like(time)
        if cdisp_widths:
            cwidth = cdisp_widths[int_left:int_right]

        # If a moved integration was in this data, report it.
        moved_int = []
        if moved_ints:
            moved_int = sorted([j for j in moved_ints if (j >= int_left and j <= int_right)])

        # Now convert to dictionary.
        segment = {"data":data,
                   "err":err,
                   "badpixmask":badpixmask,
                   "jwstdq":jwstdq,
                   "junidq":junidq,
                   "time":time,
                   "wavelengths":wavelengths,
                   "disp":disp,
                   "cdisp":cdisp,
                   "cwidth":cwidth,
                   "insts":insts,
                   "detectors":dets,
                   "filters":filters,
                   "gratings":gratings,
                   "integrations":ints,
                   "flagged":moved_int}

        # And save that segment as a .npy file.
        np.save(os.path.join(outdir, '{}.npy'.format(outfile)),segment)

        # Advance int_left.
        int_left = int_right

def save_s4_output(oneD_spec, oneD_err, time, wav_sols, shifts,
                   xpos, ypos, widths, insts, dets, filters, gratings,
                   outfile, outdir):
    """Saves a .npy for the extracted 1D spectra.

    Args:
        oneD_spec (np.array): extracted 1D spectra.
        oneD_err (np.array): extracted uncertainties on 1D spectra.
        time (np.array): mid-exposure times for each 1D spectrum.
        wav_sols (np.array): wavelength solutions for the 1D spectra.
        shifts (np.array): cross-correlation shfits for 1D spectra.
        xpos (np.array): dispersion positions for trace.
        ypos (np.array): cross-dispersion positions for trace.
        widths (np.array): cross-dispersion widths for trace.
        insts (list of str): instruments used in this observation.
        dets (list of str): detectors used in this observation.
        filters (list of str): filters used in this observation.
        gratings (list of str): gratings used in this observation.
        outfile (str): name of the output file.
        outdir (str): directory to which the .nc file will be saved to.
    """
    # First, patch for missing shifts.
    if len(shifts) == 0:
        shifts = [0 for i in time]

    # Convert to dictionary.
    spectra = {"spectrum":oneD_spec,
               "err":oneD_err,
               "waves":wav_sols,
               "shifts":shifts,
               "xpos":xpos,
               "ypos":ypos,
               "widths":widths,
               "time":time,
               "insts":insts,
               "detectors":dets,
               "filters":filters,
               "gratings":gratings}

    # And save that segment as a .npy file.
    np.save(os.path.join(outdir, '{}.npy'.format(outfile)),spectra)

def read_one_spec(file):
    """Read one 1D spectra .npy file and return its attributes.

    Args:
        file (str): path to the .npy file you want to read out.

    Returns:
        np.array, np.array, np.array, np.array, np.array, np.array, np.array, \
        np.array, list: the spectrum, uncertainties, wavelength solutions, \
        alignment shifts, dispersion/cross-dispersion positions and widths, \
        times of mid-exposure for each spectrum, and the observing details \
        which are instrument, detector, filter, and grating.
    """
    spectra = np.load(file,allow_pickle=True).item()
    spectrum = spectra["spectrum"]
    err = spectra["err"]
    waves = spectra["waves"]
    shifts = spectra["shifts"]
    xpos = spectra["xpos"]
    ypos = spectra["ypos"]
    widths = spectra["widths"]
    time = spectra["time"]
    insts = spectra["insts"]
    dets = spectra["detectors"]
    filters = spectra["filters"]
    gratings = spectra["gratings"]
    return spectrum, err, waves, shifts, xpos, ypos, widths, time,\
        insts, dets, filters, gratings

def stitch_spectra(files, detector_method, time_step, verbose):
    """Reads in *1Dspec.npy files and concatenates them if needed.

    Args:
        files (list of str): paths to all *1Dspec.npy files you intend to process.
        detector_method (str): if not None, how to handle when 1D spectra from \
        multiple detectors are found.
        time_step (bool): whether to report timing with tqdm.
        verbose (int): from 0 to 2. How much logging to do.

    Returns:
        dict: 1D spectra dict containing spectra.
    """
    # Log.
    if verbose >= 1:
        print("Stitching data files together for post-processing...")
    
    if verbose == 2:
        print("Will parse spectra from the following files:")
        for i, f in enumerate(files):
            print(i, f)

    # If there is just one file, we can take it as a dict and adjust it to have the detector dim.
    if len(files) == 1:
        if verbose >= 1:
            print("Reading out one 1D spectrum...")
        spectrum, err, waves, shifts, xpos, ypos, widths, time, \
             insts, dets, filters, gratings = read_one_spec(files[0])
        # Simply bundle it together in a dictionary.
        output = {}
        output["spectrum"] = [spectrum,]
        output["errors"] = [err,]
        output["waves"] = [waves,]
        output["shifts"] = [shifts,]
        output["xpos"] = [xpos,]
        output["ypos"] = [ypos,]
        output["widths"] = [widths,]
        output["time"] = [time,]
        output["insts"] = [insts,]
        output["detectors"] = [dets,]
        output["filters"] = [filters,]
        output["gratings"] = [gratings,]

    else:
        # Initialize some empty lists.
        spectra, errors, waves, shifts, xpos, ypos, widths = [], [], [], [], [], [], [] # the data_vars
        time = [] # the coords
        insts, dets, filters, gratings = [], [], [], [], [] # the attributes

        # Read in each file.
        for file in tqdm(files,
                        desc = 'Parsing spectral files...',
                        disable=(not time_step)):
            spectrum_i, err_i, waves_i, shifts_i, xpos_i, ypos_i, widths_i, time_i, \
                insts_i, dets_i, filters_i, gratings_i = read_one_spec(file)
            spectra.append(spectrum_i)
            errors.append(err_i)
            waves.append(waves_i)
            shifts.append(shifts_i)
            xpos.append(xpos_i)
            ypos.append(ypos_i)
            widths.append(widths_i)
            time.append(time_i)
            insts.append(insts_i)
            dets.append(dets_i)
            filters.append(filters_i)
            gratings.append(gratings_i)

        # Now check instructions.
        if detector_method == "parallel":
            # We do not sum anything. Instead, dict needs to contain each spectrum separately.
            if verbose >= 1:
                print("Parallelising multiple spectra...")

            # Simply bundle it together in a dictionary.
            output = {}
            output["spectrum"] = spectra
            output["errors"] = errors
            output["waves"] = waves
            output["shifts"] = shifts
            output["xpos"] = xpos
            output["ypos"] = ypos
            output["widths"] = widths
            output["time"] = time
            output["insts"] = insts
            output["detectors"] = dets
            output["filters"] = filters
            output["gratings"] = gratings
        
        elif detector_method == "join":
            # Check which detectors you are trying to join and warn the user about heinous combos.
            print("Joining multiple spectra...")
            gratings = [x for x in output["gratings"]]
            if any([gratings[0] != grating for grating in gratings]): # if any grating shows up that does not match the first one
                if verbose >= 1:
                    print("Warning: I noticed you are trying to stitch together files that use different dispering elements. " \
                          "While I commend your bravery, please note that the ''join'' method of combining files " \
                          "was intended only for single dispersers which span multiple detectors (e.g. G395H) " \
                          "and the correct method for treating multiple dispersers is ''parallel''. " \
                          "If you are not using limb darkening models like ExoTiC-LD, this should not crash the code." \
                          "(It will still cause some creative and surprising behavior though.)" \
                          "If you are using limb darkening models, please relaunch Stage 5 with the ''detectors'' keyword set to ''parallel''.")
            
            # For every timestamp, we have to concatenate each 1D spectrum together as well as the wavelength solution.
            con_spec, con_err, con_waves = [], [], []
            time = np.array(time)
            time = np.median(time,axis=0) # should collapse time to roughly the mid-exposure times for all 1D spectra being joined
            shifts = np.array(shifts)
            shifts = np.median(shifts,axis=0) # should be approximately the same since the detectors are parallel 

            # FIX: these should not be the same but i'll figure it out later.
            xpos = np.array(xpos)
            xpos = np.median(xpos,axis=0) # should be approximately the same since the detectors are parallel 
            ypos = np.array(ypos)
            ypos = np.median(ypos,axis=0) # should be approximately the same since the detectors are parallel 
            widths = np.array(widths)
            widths = np.median(widths,axis=0) # should be approximately the same since the detectors are parallel 

            for i in range(time.shape[0]):
                # At every time stamp, grab each spectrum's 1D spec.
                spec_i = spectra[0][i,:]
                err_i = errors[0][i,:]
                waves_i = waves[0][i,:]
                for j in range(1,len(spectra)):
                    spec_i = np.concatenate((spec_i,spectra[j][i,:]))
                    err_i = np.concatenate((err_i,errors[j][i,:]))
                    waves_i = np.concatenate((waves_i,waves[j][i,:]))
                con_spec.append(spec_i)
                con_err.append(err_i)
                con_waves.append(waves_i)

            # Simply bundle it together in a dictionary.
            output = {}
            output["spectrum"] = [con_spec,]
            output["errors"] = [con_err,]
            output["waves"] = [con_waves,]
            output["shifts"] = [shifts,]
            output["xpos"] = [xpos,]
            output["ypos"] = [ypos,]
            output["widths"] = [widths,]
            output["time"] = [time,]
            output["insts"] = [insts[0],]
            output["detectors"] = [dets[0],]
            output["filters"] = [filters[0],]
            output["gratings"] = [gratings[0],]

    return output

def read_one_lc(file):
    """Read one light curve .npy file and return its attributes.

    Args:
        file (str): path to the .npy file you want to read out.

    Returns:
        dict: a dictionary containing the extracted 1D spectrum,
        uncertainties, wavelength solutions, alignment shifts, times of
        mid-exposure for each spectrum, and the observing details which are
        instrument, detector, filter, and grating.
    """
    curves = np.load(file,allow_pickle=True)
    
    return curves

def save_s5_output(planets, planets_err, flares, flares_err,
                   systematics, systematics_err, ld, ld_err,
                   time, light_curve, errors, wavelength,
                   outfile, outdir):
    """Writes out the results of a fit to a .npy file.

    Args:
        planets (dict): every fitted planet.
        planets_err (dict): every fitted planet's uncertainties.
        flares (dict): every fitted flare.
        flares_err (dict): every fitted flare's uncertainties.
        systematics (dict): fitted systematics models.
        systematics_err (dict): uncertainties on systematics models.
        ld (dict): fitted limb darkening model.
        ld_err (dict): uncertainties on limb darkening model.
        time (np.array): timestamps for each flux point.
        light_curve (np.array): flux at each point in time.
        errors (np.array): uncertainty at each point in time.
        wavelength (float): central wavelength for this data.
        outfile (str): name to give the saved file.
        outdir (str): where to save the file to.
    """
    # Dict everything together.
    output = {"planets":planets,
              "planet_errs":planets_err,
              "flares":flares,
              "flare_errs":flares_err,
              "systematics":systematics,
              "systematic_errs":systematics_err,
              "ld":ld,
              "ld_err":ld_err,
              "time":time,
              "light_curve":light_curve,
              "errors":errors,
              "wavelength":wavelength}
    
    filename = (os.path.join(outdir, '{}.npy'.format(outfile)))
    np.save(filename,output)
