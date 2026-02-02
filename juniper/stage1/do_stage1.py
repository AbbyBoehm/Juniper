import os
from tqdm import tqdm

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.config.translate_config import s1_to_pipeline, s1_to_glbs, s1_to_miribckg, s1_to_NSClean
from juniper.stage1 import group_level_bckg_sub, miri_bckg_sub, wrap_stage1jwst, NSClean

def do_stage1(filepaths, outfiles, outdir, steps, plot_dir):
    """Performs Stage 1 calibration on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct. The files must be of type *_uncal.fits.
        outfiles (list): lst of str. Names to give to the calibrated files.
        outdir (str): location of where to save the calibrated files to.
        steps (dict): instructions on how to run this stage of the pipeline. Loaded from the Stage 1 .berry files.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 1 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 1 will operate and output to the following files:")
        for i, f in enumerate(filepaths):
            print(i, f, "->", outfiles[i])
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_plots = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Start iterating.
    for filepath, outfile in tqdm(zip(filepaths, outfiles),
                                  desc='Processing Stage 1...',
                                  disable=(not time_step)):
        # Build the pipeline dictionary.
        s1_pipeline = s1_to_pipeline(steps)
        # Wrap the first steps of Detector1Pipeline.
        datamodel = wrap_stage1jwst.wrap_front_end(filepath, s1_pipeline)

        # Perform group-level background subtraction.
        if steps["do_glbs"]:
            s1_glbs = s1_to_glbs(steps)
            datamodel = group_level_bckg_sub.glbs(datamodel, s1_glbs, plot_dir, outfile)

        # Perform MIRI LRS background subtraction.
        if steps["do_miribckg"]:
            s1_miribckg = s1_to_miribckg(steps)
            datamodel = miri_bckg_sub.miribckg(datamodel, dict(s1_pipeline), s1_miribckg, plot_dir, outfile)

        # Wrap the last steps of Detector1Pipeline.
        result = wrap_stage1jwst.wrap_back_end(datamodel, s1_pipeline, outfile, outdir)

        # Perform NSClean background subtraction.
        if steps["do_NSClean"]:
            s1_NSClean = s1_to_NSClean(steps)
            NSClean.NSClean(filepath, s1_NSClean, plot_dir)
        
        if steps["verbose"] == 2:
            print("One iteration complete. Output saved in", outdir, "as file name {}.fits".format(outfile))
    
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 1 is complete.")