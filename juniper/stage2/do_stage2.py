import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from astropy.io import fits
from jwst import datamodels as dm

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.config.translate_config import s2_to_pipeline, s2_clean_dict
from juniper.stage2 import wrap_stage2jwst, correct_curvature, miri_wavelength_map, truncate_array

def do_stage2(filepaths, outfiles, outdir, steps, plot_dir):
    """Performs Stage 2 calibration on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct.
        The files must be of type *_rateints.fits.
        outfiles (list): lst of str. Names to give to the calibrated files.
        outdir (str): location of where to save the calibrated files to.
        steps (dict): instructions on how to run this stage of the pipeline.
        Loaded from the Stage 2 .berry files.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 2 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 2 will operate and output to the following files:")
        for i, f in enumerate(filepaths):
            print(i, f, "->", outfiles[i]+".fits")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_plots = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist, and plot directory if you want plots.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if (any([save_step, save_plots]) and not os.path.exists(plot_dir)):
        os.makedirs(plot_dir)

    i = 0
    # Start iterating.
    for filepath, outfile in tqdm(zip(filepaths, outfiles),
                                  total=len(outfiles),
                                  desc='Processing Stage 2...',
                                  disable=(not time_step)):
        # Build the pipeline dictionary.
        s2_pipeline = s2_to_pipeline(steps)

        # Check observing mode and remove unneeded tags.
        with fits.open(filepath) as f:
            mode = f[0].header['INSTRUME']
            if steps["verbose"] >= 1:
                print("Operating on mode {}. Adjusting input dictionary to match expected keys.".format(mode))
            s2_pipeline = s2_clean_dict(s2_pipeline, mode)
        # Process Spec2Pipeline.
        wrap_stage2jwst.wrap(filepath, outfile, outdir, s2_pipeline)

        # Then curve-correct, if necessary.
        s2_curvecorrect = {}
        for key in ("verbose","show_plots","save_plots"):
            s2_curvecorrect[key] = steps[key]
        s2_curvecorrect["diagnostic_plots"] = plot_dir
        if (steps["do_correction"] and "NIRSPEC".casefold() in mode.casefold()):
            correct_curvature.correct_curvature(outfile, outdir, i, s2_curvecorrect)
        
        # If MIRI, assign wavelength solution.
        s2_wavelengthmap = {}
        for key in ("verbose","show_plots","save_plots"):
            s2_wavelengthmap[key] = steps[key]
        if (steps["do_wavemap"] and "MIRI".casefold() in mode.casefold()):
            miri_wavelength_map.wavemap(outfile, outdir, filepath, s2_wavelengthmap)

        # If desired, truncate the array.
        s2_truncate = {}
        for key in ("verbose","show_plots","save_plots","keep_rows","keep_cols"):
            s2_truncate[key] = steps[key]
        s2_truncate["diagnostic_plots"] = plot_dir
        if steps["do_truncate"]:
            truncate_array.truncate(outfile, outdir, s2_truncate)

        # If asked, create a plot comparing the input and output 0th integration.
        if (plot_step or save_step):
            # Load integration 0 of the input *rateints.fits file
            with dm.open(filepath) as f1, dm.open(os.path.join(outdir,outfiles[i]+".fits")) as f2:
                data1, data2 = f1.data, f2.data

                data1[np.isnan(data1)] = 0
                data2[np.isnan(data2)] = 0

                vmin1, vmax1 = np.percentile(data1,q=5), np.percentile(data1,q=95)
                vmin2, vmax2 = np.percentile(data2,q=5), np.percentile(data2,q=95)
                lin_threshold = 0.1
                symlog_norm_1 = colors.SymLogNorm(linthresh=lin_threshold, 
                                                  linscale=1, 
                                                  vmin=vmin1,
                                                  vmax=vmax1,
                                                  base=10)
                symlog_norm_2 = colors.SymLogNorm(linthresh=lin_threshold, 
                                                  linscale=1, 
                                                  vmin=vmin2,
                                                  vmax=vmax2,
                                                  base=10)
                
                fig, ax = plt.subplots(figsize=(20,8),nrows=2)
                im = ax[0].imshow(data1[0,:,:],cmap='viridis',origin='lower',
                                  norm=symlog_norm_1,aspect='auto')
                ax[1].imshow(data2[0,:,:],cmap='viridis',origin='lower',
                             norm=symlog_norm_2,aspect='auto')
                cbar = plt.colorbar(mappable=im,orientation='horizontal',
                                    aspect=40)
                cbar.set_label("Calibrated Flux [DN]")
                ax[0].set_title("Rateints Integration 0")
                ax[1].set_title("Calints Integration 0")

                if save_step:
                    plt.savefig(os.path.join(plot_dir,"S2_{}_rateints0_v_calints0_f{}.png".format(outfile,i)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()

        
        if steps["verbose"] == 2:
            print("One iteration complete. Output saved in", outdir, "as file name {}".format(outfile))
        
        i += 1
    
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 2 is complete.")