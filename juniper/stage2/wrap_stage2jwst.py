import time

from astropy.io import fits
from jwst.pipeline import Spec2Pipeline

from juniper.util.diagnostics import timer

def wrap(filepath,outfile,outdir,inpt_dict):
    """Wrapper for jwst Spec2Pipeline.

    Args:
        filepath (str): a path to the *rateints.fits file you want to operate on.
        outfile (str): what to rename the output files to. Can be None to keep the default name.
        outdir (str): where to save the output *calints.fits files.
        inpt_dict (dict): A dictionary containing instructions for all stages of the Spec2Pipeline.
    """
    # Time this step if asked.
    if inpt_dict["verbose"] >= 1:
        t0 = time.time()

    # Copy dict and modify it.
    s2_steps = inpt_dict.copy()
    # Delete entries related to verbose, show_plots, and save_plots.
    for key in ("verbose","show_plots","save_plots"):
        s2_steps.pop(key, None)
    
    # Process Spec2Pipeline.
    result = Spec2Pipeline.call(filepath, output_file=outfile, output_dir=outdir,
                                steps=s2_steps)
    
    if inpt_dict["verbose"] >= 1:
        timer(time.time()-t0,None,None,None)