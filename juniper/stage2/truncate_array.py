import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from astropy.io import fits

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import img

def truncate(outfile, outdir, inpt_dict):
    """Trims array down to the rows/cols the user wants to keep.

    Args:
        outfile (str): The name of the file we are checking, sans "_calints.fits".
        outdir (str): The directory where the file can be found.
        inpt_dict (dict): A dictionary containing instructions for performing this step.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Truncating array...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Set up the output file name.
    output_file = os.path.join(outdir, outfile+".fits")

    with fits.open(output_file, mode="update") as fits_file:
        # Need to update arrays to be truncated.
        r1, r2 = inpt_dict['keep_rows']
        c1, c2 = inpt_dict['keep_cols']
        for header in ('SCI','ERR','DQ','WAVELENGTH','VAR_POISSON','VAR_RNOISE'):
            try:
                fits_file[header].data = fits_file[header].data[:,r1:r2,c1:c2]
            except KeyError:
                if inpt_dict["verbose"] >= 1:
                    print("Dataset does not contain header {}.".format(header))
            except IndexError:
                if inpt_dict["verbose"] >= 1:
                    print("Dataset with header {} has dims ".format(header),fits_file[header].data.shape,
                          "incompatible with truncation to rows ({},{}) and cols ({},{}).".format(r1,r2,c1,c2))

        # All modified headers get written out.
        fits_file.writeto(output_file, overwrite=True)

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Arrays truncated.")

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
