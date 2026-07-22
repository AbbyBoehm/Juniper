import os
from tqdm import tqdm
import numpy as np

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_npys, save_s4_output
from juniper.stage4 import clean_signal, extract_1D, extract_photseries, align_spec, decorrelate_photseries, plot_signal_gif

def do_stage4(filepaths, outfile, outdir, steps, plot_dir):
    """Performs Stage 4 extraction on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to extract
        from. The files must be of type *_reduced.nc
        outfile (str): name to give to the extracted signal time-series file.
        outdir (str): location of where to save the signal time-series file to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 4 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 4 will operate on the following files:")
        for i, f in enumerate(filepaths):
            print(i, f)
        print("Output will be saved to {}.nc.".format(outfile))
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_ints = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Put the plot directory into the inpt_dict and create it.
    steps["plot_dir"] = plot_dir
    if (not os.path.exists(plot_dir) and any((save_step, save_ints))):
        os.makedirs(plot_dir)

    # Open all files and stitch them together.
    segments = stitch_npys(filepaths,
                           time_step=time_step,
                           verbose=steps["verbose"])
    
    # Check type and assign flag.
    exposure_type = "spectroscopic"
    if np.median(segments["wavelengths"]) == np.std(segments["wavelengths"]) == 0:
        # This is the placeholder that is assigned in Stage 3 when the data is photometric.
        exposure_type = "photometric"
    if steps["verbose"] == 2:
        print("Data registered as type:",exposure_type)
    
    # Need to track xpos, ypos, widths.
    xpos, ypos, widths = (np.array(segments["disp"]),
                          np.array(segments["cdisp"]),
                          np.array(segments["cwidth"]),)
    
    if exposure_type == "spectroscopic":
        # Extract 1D spectra.
        signal_tseries, signal_err, wav_sols = extract_1D.extract(segments, steps)
    elif exposure_type == "photometric":
        # Extract photometric time-series.
        signal_tseries, signal_err = extract_photseries.extract(segments, steps)
        wav_sols = np.zeros_like(signal_tseries)

    # Kick unwanted integrations.
    bad_frames = []
    if steps["s3_kick_ints"]:
        if steps["verbose"] >= 1:
            print("Kicking frames flagged by S3 for drift from signal time-series...")
        bad_frames = [j for j in segments["flagged"][0]]
    else:
        if steps["verbose"] >= 1:
            print("Frames flagged for motion in S3 will NOT be kicked in this run.")
    if steps["s4_trim_ints"]:
        if steps["verbose"] >= 1:
            print(f"Trimming integrations {steps['s4_trim_ints']} from signal time-series...")
        for trim_ints in steps["s4_trim_ints"]:
            trim = range(trim_ints[0],trim_ints[1]+1)
            for i in [j for j in trim if j not in bad_frames]:
                bad_frames.append(i)
    # Now that all bad frames are found, kick them.
    signal_tseries = np.delete(signal_tseries, bad_frames, axis=0)
    signal_err = np.delete(signal_err, bad_frames, axis=0)
    wav_sols = np.delete(wav_sols, bad_frames, axis=0)
    xpos = np.delete(xpos, bad_frames)
    ypos = np.delete(ypos, bad_frames)
    widths = np.delete(widths, bad_frames)
    time = np.delete(segments["time"], bad_frames, axis=0)
    if (steps["verbose"] >= 1 and (steps["s3_kick_ints"] or steps["s4_trim_ints"])):
        print("{} integrations deleted from time-series.".format(len(bad_frames)))

    # Align spectra if applicable.
    shifts = []
    if steps["align"] and exposure_type == "spectroscopic":
        signal_tseries, signal_err, shifts = align_spec.align(signal_tseries, signal_err,
                                                              wav_sols, time, steps)
        
    # Decorrelate photometric time-series if applicable.
    if steps["decorrelate"] and exposure_type == "photometric":
        signal_tseries, signal_err, shifts = decorrelate_photseries.decorrelate(signal_tseries,
                                                                                segments['disp'],
                                                                                segments['cdisp'],
                                                                                segments['cwidth'],
                                                                                time, steps)
        
    # Clean spectra.
    if steps["sigma"]:
        if exposure_type == "spectroscopic":
            signal_tseries = clean_signal.clean_spec(signal_tseries, steps)
        if exposure_type == "photometric":
            signal_tseries == clean_signal.clean_photseries(signal_tseries, time, steps)

    # Make diagnostic static plots.
    if (plot_step or save_step):
        if exposure_type == "spectroscopic":
            plot_signal_gif.make_stack(signal_tseries,wav_sols,time,steps)
            plot_signal_gif.make_err_median(signal_tseries,wav_sols,signal_err,steps)
            plot_signal_gif.make_wlc(signal_tseries,wav_sols,time,steps)
        if exposure_type == "photometric":
            print("GO INSTALL SOME NICE PHOT PLOTS PLEASE :3")
    # Make diagnostic gif.
    if (plot_ints or save_ints):
        if exposure_type == "spectroscopic":
            plot_signal_gif.make_gif(signal_tseries,wav_sols,time,steps)
        if exposure_type == "photometric":
            print("GO INSTALL SOME NICE PHOT PLOTS PLEASE :3")

    # Save everything out.
    save_s4_output(signal_tseries, signal_err, time, wav_sols,
                   shifts, xpos, ypos, widths,
                   segments["insts"], segments["detectors"],
                   segments["filters"], segments["gratings"],
                   outfile, outdir)

    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 4 is complete.")