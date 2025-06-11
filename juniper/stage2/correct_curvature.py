import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from astropy.io import fits

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import img

def correct_curvature(outfile, outdir, inpt_dict):
    """Checks if the file needs its curvature corrected, and if it does, corrects it.

    Args:
        outfile (str): The name of the file we are checking, sans "_calints.fits".
        outdir (str): The directory where the file can be found.
        inpt_dict (dict): A dictionary containing instructions for performing this step.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Curvature correction processing...")
    
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
    with fits.open(output_file) as file:
        grating = file[0].header['GRATING']
        if grating in ("G395M","G395H"):
            if inpt_dict["verbose"] == 2:
                print("{} grating detected, correcting for trace curvature...".format(grating))
            shifted_data, shifted_wvs = fix_curvature(file['SCI'].data,
                                                      file['WAVELENGTH'].data,
                                                      timer=[time_step,time_ints],
                                                      show=[plot_step,plot_ints],
                                                      save=[save_step,save_ints],
                                                      verbose=inpt_dict["verbose"],
                                                      outdir=inpt_dict["diagnostic_plots"],
                                                      outfile=outfile)
            write_curve_fixed_file(output_file, shifted_data, shifted_wvs)
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Curvature correction complete.")

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

def fix_curvature(data, wvs, timer, show, save, verbose, outdir, outfile):
    """Corrects trace curvature in given array. Adapted in part from Eureka!

    Args:
        data (np.array): 3D rateints data.
        wvs (np.array): 3D wavelength solution.
        timer (list): bool, bool. Respectively whether to time the whole step
        and whether to time corrections to each frame.
        show (list): bool, bool. Respectively whether to plot an overall
        diagnostic plot and whether to plot diagnostics for every frame.
        save (list): bool, bool. Respectively whether to save an overall
        diagnostic plot and whether to save diagnostics for every frame.
        verbose (int): from 0 to 2. How much logging this step should do.
        outdir (str): where to save output plots to, if applicable.
        outfile (str): helps keep diagnostic plots distinct.

    Returns:
        np.array, np.arrray: corrected data and wvs arrays.
    """
    # Time this step if asked.
    time_step, time_ints = timer
    plot_step, plot_ints = show
    save_step, save_ints = save
    
    # Use the median frame to determine needed rolls.
    medframe = np.median(data,axis=0)
    medframe[np.isnan(medframe)] = 0 # get rid of nans because they upset the roller.
    rolls = get_rolls(medframe) # get the rolls needed to correct the framese.

    if (plot_step or save_step):
        plt.figure(figsize=(7,4))
        plt.plot(rolls)
        plt.xlabel("column position [pixels]")
        plt.ylabel("roll [pixels]")
        plt.title("Rolls needed to correct frames")
        plt.ylim(-13, 13)
        if save_step:
            plt.savefig(os.path.join(outdir,"S2_{}_curvature_corrections.png".format(outfile)))
        if plot_step:
            plt.show(block=True)
        plt.close()

    # There is only one wavelength frame, so roll it by the median rolls in time.
    shifted_wvs = roll_one_frame(wvs, rolls)

    # Then for each frame in data, need to roll it.
    shifted_data = np.empty_like(data)
    for i in tqdm(range(data.shape[0]),
                  desc='Correcting curvature in each frame...',
                  disable=(not time_ints)):
        shifted_data[i,:,:] = roll_one_frame(data[i,:,:], rolls)
        if (plot_step or save_step):
            if ((not plot_ints and i == 0) or plot_ints or save_ints): # either if just plot/save the first frame, or plot/save all ints
                fig, ax, im = img(shifted_data[i,:,:],
                                  aspect=5,
                                  title="Rolled frame {}".format(i),
                                  norm='log',
                                  vmin=0.01,
                                  vmax=100,
                                  verbose=verbose)
                if save_step:
                    plt.savefig(os.path.join(outdir,"S2_{}_corrected_frame{}.png".format(outfile,i)))
                if plot_step:
                    plt.show(block=True)
                plt.close()    
    return shifted_data, shifted_wvs

def roll_one_frame(frame, rolls):
    """Roll one frame into alignment.

    Args:
        frame (np.array): one frame from *calints.fits.
        rolls (list): how many pixels by which to roll each column in the frame.

    Returns:
        np.array: frame rolled into alignment.
    """
    retain_last_roll = 0
    for j, roll in enumerate(rolls):
        if abs(roll) > 20:
            # If an outlier roll is found, we use the same roll that we used the last time a roll succeeded.
            roll = retain_last_roll
        frame[:,j] = np.roll(frame[:,j], int(roll))
        retain_last_roll = roll
    return frame

def get_rolls(frame):
    """Determine the rolls needed to straighten the trace using the median frame.
    Credit Eureka! S3 straighten.py code.

    Args:
        frame (np.array): median frame used to measure the rolls.

    Returns:
        list: int values used to roll traces into alignment.
    """
    pix_centers = np.arange(frame.shape[0]) + 0.5
    COMs = signal.medfilt((np.sum(pix_centers[:,np.newaxis]*np.abs(frame),axis=0)/np.sum(np.abs(frame),axis=0)),7)
    integer_COMs = np.around(COMs - 0.5).astype(int)
    new_center = int(frame.shape[0]/2) - 1
    rolls = new_center - integer_COMs
    rolls[COMs<0] = 0
    rolls[COMs>frame.shape[0]] = 0
    rolls = signal.medfilt(rolls,41)
    return rolls

def write_curve_fixed_file(output_file, shifted_data, shifted_wvs):
    """Write curvature-corrected file.

    Args:
        output_file (str): name of the *calints.fits file we just rolled
        and are going to overwrite.
        shifted_data (np.array): 3D data that has been rolled.
        shifted_wvs (np.array): 3D wavelength solution that has been rolled.
    """
    with fits.open(output_file, mode="update") as fits_file:
        # Need to update data and wavelength attributes to be rotated arrays.
        fits_file['SCI'].data = shifted_data
        fits_file['WAVELENGTH'].data = shifted_wvs

        # All modified headers get written out.
        fits_file.writeto(output_file, overwrite=True)