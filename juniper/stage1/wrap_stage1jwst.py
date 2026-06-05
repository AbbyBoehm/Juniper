import time

from jwst.pipeline import Detector1Pipeline

from juniper.util.diagnostics import timer

def wrap_front_end(filepath, inpt_dict):
    '''
    Wrapper for all jwst Detector1Pipeline steps before ramp_fit.
    
    :param filepath: str. A path to the *uncal.fits file you want to operate on.
    :param inpt_dict: dict. A dictionary containing instructions for all stages of the Detector1Pipeline. The ramp_fit and gain_scale instructions are ignored here.
    :return: JWST datamodel produced by Detector1Pipeline.
    '''
    # Time step, if asked.
    if inpt_dict["verbose"] >= 1:
        t0 = time.time()
    
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("jwst pipeline Stage 1 front end processing...")
        
    # Copy dict and modify it.
    front_end_steps = inpt_dict.copy()
    for step in ("ramp_fit","gain_scale"):
        front_end_steps[step] = {"skip":True}
    # Delete entries related to verbose, show_plots, and save_plots.
    for key in ("verbose","show_plots","save_plots"):
        front_end_steps.pop(key, None)

    if inpt_dict["verbose"] == 2:
        print("Detector1Pipeline front end running with the following arguments:")
        for key in list(front_end_steps.keys()):
            print(key, front_end_steps[key])
    
    # Process Detector1Pipeline front end.
    result = Detector1Pipeline.call(filepath,
                                    steps=front_end_steps)
    
    # Report time, if asked.
    if inpt_dict["verbose"] >= 1:
        timer(time.time()-t0,None,None,None)
    return result

def wrap_back_end(datamodel, inpt_dict, outfile, outdir):
    '''
    Wrapper for all jwst Detector1Pipeline steps including and after ramp_fit.
    
    :param datamodel: JWST datamodel. A datamodel containing attribute .data, which is an np array of shape nints x ngroups x nrows x ncols, produced during wrap_front_end.
    :param inpt_dict: dict. A dictionary containing instructions for Detector1Pipeline. Only the ramp_fit and gain_scale entries are read here.
    :param outfile: str. Name of the output *rateints.fits file.
    :param outdir: str. Relative or absolute path to where the outfile will be saved.
    :return: outfile saved to outdir. Routine returns no callables.
    '''
    # Time step, if asked.
    if inpt_dict["verbose"] >= 1:
        t0 = time.time()

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("jwst pipeline Stage 1 back end processing...")
    
    # Copy dict and modify it.
    back_end_steps = inpt_dict.copy()
    for step in ("group_scale","dq_init","saturation","superbias",
                 "refpix","linearity","dark_current","jump","clean_flicker_noise",
                 "persistence","emicorr","firstframe","lastframe",
                 "reset","rscd","charge_migration"):
        back_end_steps[step] = {"skip":True}
    # Delete entries related to verbose, show_plots, and save_plots.
    for key in ("verbose","show_plots","save_plots"):
        back_end_steps.pop(key, None)
    
    if inpt_dict["verbose"] == 2:
        print("Detector1Pipeline back end running with the following arguments:")
        for key in list(back_end_steps.keys()):
            print(key, back_end_steps[key])
    
    # Process Detector1Pipeline back end.
    result = Detector1Pipeline.call(datamodel,
                                    output_file=outfile,
                                    output_dir=outdir,
                                    steps=back_end_steps)
    
    if inpt_dict["verbose"] >= 1:
        timer(time.time()-t0,None,None,None)
    
    return result