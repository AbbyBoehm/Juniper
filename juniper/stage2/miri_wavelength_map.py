import os
import time
import numpy as np

from astropy.io import fits
from jwst import datamodels
from jwst.pipeline.calwebb_spec2 import assign_wcs_step, srctype_step

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer


def wavemap(outfile, outdir, inpt_rateints, inpt_dict):
    """Builds a wavelength solution using wcs and srctype.

    Args:
        outfile (str): The name of the file we are assigning a wavelength
        solution to, sans "_calints.fits".
        outdir (str): The directory where the file can be found.
        inpt_rateints (str): The name of the file we use to assign a wavelength
        solution to, sans "_rateints.fits".
        inpt_dict (dict): A dictionary containing instructions for performing this step.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Assigning wavelength solution to MIRI image...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

        # Set up the output file name.
    output_file = os.path.join(outdir, outfile+".fits")

    # Open the datamodel to grab the wcs info.
    with datamodels.open(inpt_rateints) as datamodel:
        datamodel = assign_wcs_step.AssignWcsStep().call(datamodel)
        datamodel = srctype_step.SourceTypeStep().call(datamodel)

        rows, cols = np.mgrid[0:datamodel.data.shape[1],0:datamodel.data.shape[2]]
        wavelengths = datamodel.meta.wcs(cols.ravel(),rows.ravel())[-1].reshape((datamodel.data.shape[1],
                                                                                 datamodel.data.shape[2]))

    with fits.open(output_file, mode="update") as fits_file:
        if fits_file[0].header['INSTRUME'] == 'MIRI':
            # Give the fits file a new attribute containing the wavelengths object
            fits_file['WAVELENGTH'].data = wavelengths
            if inpt_dict["verbose"] >= 1:
                print("MIRI file recognized: wavelength solution written out to *_calints.fits file.")
        else:
            # Non-MIRI file types already have a wavelength map ready to go.
            if inpt_dict["verbose"] >= 1:
                print("File not recognized as MIRI: wavelength solution already written.")

        # All modified headers get written out.
        fits_file.writeto(output_file, overwrite=True)
    
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Wavelength mapping step complete.")

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
