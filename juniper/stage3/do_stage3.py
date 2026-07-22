import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_files, save_s3_output
from juniper.util.plotting import aspect_handler
from juniper.stage3 import reject_time, reject_space, reject_flagged, track_motion, subtract_background

def do_stage3(filepaths, outfiles, outdir, steps, plot_dir):
    """Performs Stage 3 reduction on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct.
        The files must be of type *_calints.fits.
        outfiles (list): lst of str. Names to give to the reduced files.
        outdir (str): location of where to save the reduced files to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 3 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 3 will operate and output to the following files:")
        for i, f in enumerate(filepaths):
            print(i, f, "->", outfiles[i]+".fits")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_ints = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Add the plot_dir to the stes.
    steps["diagnostic_plots"] = plot_dir

    # Open all files and stitch them together.
    segments = stitch_files(filepaths,
                            time_step=time_step,
                            verbose=steps["verbose"])
    
    if plot_step or save_step:
        raw_f0 = np.copy(segments["data"][0,:,:])
    if plot_ints or save_ints:
        raw_fs = np.copy(segments["data"][:,:,:])
    
    # Mask data flags.
    if steps["reject_flagged"]:
        if steps["verbose"] >= 1:
            print("JWST flags will be used for this run. Output data quality array will include JWST flag information.")
        segments = reject_flagged.mask_flags(segments, steps)
        modified_by_reject_flagged = np.count_nonzero(segments["jwstdq"])

    # Alternatively, if JWST flags are not of interest, replace them.
    if not steps["reject_flagged"]:
        if steps["verbose"] >= 1:
            print("JWST flags will be ignored for this run. Output data quality array will exclude JWST flag information.")
        segments["jwstdq"] = np.zeros_like(segments["jwstdq"])
        modified_by_reject_flagged = 0
        
    # Reject outliers in time.
    if steps["reject_time"]:
        if steps["time_method"] == "fixed":
            segments = reject_time.iterate_fixed(segments, steps)

        elif steps["time_method"] == "free":
            segments = reject_time.iterate_free(segments, steps)

        else:
            if steps["verbose"] >= 1:
                print("Input time rejection method '{}' was not recognized! Please supply method 'fixed' or 'free'.".format(steps["time_method"]))
                print("Skipping time rejection...")
        
    # Reject outliers in space.
    if steps["reject_space"]:
        if steps["space_method"] == "led":
            segments = reject_space.led(segments, steps)
        
        elif steps["space_method"] == "smooth":
            segments = reject_space.smooth(segments, steps)
        
        else:
            if steps["verbose"] >= 1:
                print("Input spatial rejection method '{}' was not recognized! Please supply method 'led' or 'smooth'.".format(steps["space_method"]))
                print("Skipping spatial rejection...")

    # Finally, report changes if asked.
    if steps["verbose"] >= 1:
        N_pixels = 1
        for dim in np.shape(segments["junidq"]):
            N_pixels *= dim
        N_modified = np.count_nonzero(segments["junidq"])
        print("All outlier rejection processes complete.")
        print("Percentage of data modified by Juniper: {:.2f}%".format(100*N_modified/N_pixels))
        if steps["reject_flagged"]:
            print("Percentage modified by Juniper because of JWST flags: {:.2f}%".format(100*modified_by_reject_flagged/N_pixels))

    # Remove background signal.
    if steps["subtract_bckg"]:
        segments = subtract_background.subtract_background(segments, steps)

    # Track motion of the signal.
    x_pos, y_pos, signal_width, moved_ints = [],[],[],[]

    if any((steps["track_disp"],steps["track_spatial"])):
        # Confirm this data is the right kind for this form of tracking.
        segments, x_pos, y_pos, signal_width, moved_ints = track_motion.track_pos(segments, steps)

    if any((steps["track_xy"],steps["track_fwhm"])):
        # Confirm this data is the right kind for this form of tracking.
        segments, x_pos, y_pos, signal_width, moved_ints = track_motion.track_psf(segments, steps)

    # Save everything out.
    save_s3_output(segments, x_pos, y_pos, signal_width, moved_ints, outfiles, outdir)

    if plot_step or save_step:
        fig, ax = plt.subplots(2,1,figsize=(20,8),sharex=True)
        img_aspect, _ = aspect_handler(segments["data"][0,:,:])
        vmin, vmax = np.nanpercentile(segments["data"][0,:,:],q=5), np.nanpercentile(segments["data"][0,:,:],q=95)
        if vmin <= 0:
            frame1 = np.copy(segments["data"][0,:,:])
            pos = frame1[frame1>0]
            vmin, vmax = np.nanpercentile(pos[np.isfinite(pos)],q=5), np.percentile(pos[np.isfinite(pos)],q=95)
        ax[0].imshow(raw_f0,cmap='viridis',origin='lower',aspect=img_aspect,
                     vmin=vmin,vmax=vmax,norm='log')
        ax[0].set_title("Pre-correction integration 0")
        ax[1].imshow(segments["data"][0,:,:],cmap='viridis',origin='lower',aspect=img_aspect,
                     vmin=vmin,vmax=vmax,norm='log')
        ax[1].set_title("Post-correction integration 0")
        if save_step:
            plt.savefig(os.path.join(steps["diagnostic_plots"],"S3_before-after_int0.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show(block=True)
        plt.close()

    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 3 is complete.")
        