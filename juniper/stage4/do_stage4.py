import os
from tqdm import tqdm
import numpy as np

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_files, save_s4_output
from juniper.stage4 import extract_1D, align_spec, clean_spec, plot_spec_gif

def do_stage4(filepaths, outfile, outdir, steps, plot_dir):
    """Performs Stage 4 extraction on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to extract
        from. The files must be of type *_reduced.nc
        outfile (str): name to give to the extracted spectra file.
        outdir (str): location of where to save the spectra file to.
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
    # FIX : i'll figure this out later
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
    segments = stitch_files(filepaths,
                            time_step=time_step,
                            verbose=steps["verbose"])
    
    # Need to track xpos, ypos, widths.
    xpos, ypos, widths = (np.array(segments.disp.values),
                          np.array(segments.cdisp.values),
                          np.array(segments.cwidth.values),)
    
    # Extract 1D spectra.
    oneD_spec, oneD_err, wav_sols = extract_1D.extract(segments, steps)

    # Kick unwanted integrations.
    if steps["verbose"] >= 1:
        print("Kicking flagged frames from 1D spectra...")
    bad_frames = []
    if steps["s3_kick_ints"]:
        bad_frames = [j for j in segments.flagged[0]]
    if steps["trim_ints"]:
        for trim_ints in steps["trim_ints"]:
            trim = range(trim_ints[0],trim_ints[1]+1)
            for i in [j for j in trim if j not in bad_frames]:
                bad_frames.append(i)
    # Now that all bad frames are found, kick them.
    oneD_spec = np.delete(oneD_spec, bad_frames, axis=0)
    oneD_err = np.delete(oneD_err, bad_frames, axis=0)
    wav_sols = np.delete(wav_sols, bad_frames, axis=0)
    xpos = np.delete(xpos, bad_frames)
    ypos = np.delete(ypos, bad_frames)
    widths = np.delete(widths, bad_frames)
    time = np.delete(segments.time.values, bad_frames, axis=0)
    if steps["verbose"] >= 1:
        print("{} integrations deleted from spectra.".format(len(bad_frames)))

    # Align spectra.
    shifts = []
    if steps["align"]:
        oneD_spec, oneD_err, shifts = align_spec.align(oneD_spec, oneD_err,
                                                       wav_sols, time, steps)
        
    # Clean spectra.
    if steps["sigma"]:
        oneD_spec = clean_spec.clean_spec(oneD_spec, steps)

    # Make diagnostic gif.
    if (plot_step or save_step):
        plot_spec_gif.make_gif(oneD_spec,wav_sols,time,steps)

    # Save everything out.
    save_s4_output(oneD_spec, oneD_err, time, wav_sols, shifts,
                   xpos, ypos, widths, segments.details[0], outfile, outdir)

    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 4 is complete.")