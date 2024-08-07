import os
import glob
from copy import deepcopy

import numpy as np
from math import isnan
import time

from scipy.optimize import least_squares
import batman
import emcee
import matplotlib.pyplot as plt

from . import utils

def doStage5(filesdir, outdir, event_type, reject_threshold=3, raise_alarm=10,
             LSQfit_WLC={"do":True,
                         "exoplanet_params":{},
                         "systematics":(1,0),
                         "limb_darkening_model":{},
                         "fixed_param":{},
                         "priors_dict":{},
                         "priors_type":"uniform",
                         "exoticLD":{}},
             MCMCfit_WLC={"do":True,
                          "exoplanet_params":{},
                          "systematics":(1,0),
                          "limb_darkening_model":{},
                          "fixed_param":{},
                          "priors_dict":{},
                          "exoticLD":{},
                          "priors_type":"uniform",
                          "N_walkers":32,
                          "N_steps":5000,
                          "est_err":0.001},
             LSQfit_spec={"do":True,
                          "exoplanet_params":{},
                          "systematics":(1,0),
                          "limb_darkening_model":{},
                          "fixed_param":{},
                          "priors_dict":{},
                          "priors_type":"uniform",
                          "exoticLD":{}},
             MCMCfit_spec={"do":True,
                           "exoplanet_params":{},
                           "systematics":(1,0),
                           "limb_darkening_model":{},
                           "fixed_param":{},
                           "priors_dict":{},
                           "exoticLD":{},
                           "priors_type":"uniform",
                           "N_walkers":32,
                           "N_steps":5000,
                           "est_err":0.001},
             save_plots={"do":True,
                         "fit_and_residuals":True,
                         "posteriors":True,
                         "corner":True,
                         "chains":True,
                         "spectrum":True},
             reference_spectra={"reference_wavelengths":[],
                                "reference_depths":[],
                                "reference_errs":[],
                                "reference_names":[]},
             ):
    '''
    Performs Stage 5 linear least-squares and MCMC fitting of the light curves in the specified directory.
    :param filesdir: str. Directory where the WLC and SLC files you want to fit to are stored, with wavelength files as information.
    :param outdir: str. Where to save all the outputs.
    :param event_type: str. The type of event being fitted, which can be "transit" or "eclipse".
    :param LSQfit_WLC: dict of "do" bool, "exoplanet_params" dict, "systematics" tuple of float, "limb_darkening_model" dict, "fixed_param" dict, "priors_dict" dict, "priors_type" str, and "exoticLD" dict. Used to perform linear least-squares fitting on WLC to get system parameters out.
    :param MCMCit_WLC: dict of "do" bool, "exoplanet_params" dict, "systematics" tuple of float, "limb_darkening_model" dict, "fixed_param" dict, "priors_dict" dict, "priors_type" str, "exoticLD" dict, "N_walkers" int, "N_steps" int, and "est_err" float. Used to perform MCMC fitting on WLC to get system parameters out.
    :param LSQfit_spec: dict of "do" bool, "exoplanet_params" dict, "systematics" tuple of float, "limb_darkening_model" dict, "fixed_param" dict, "priors_dict" dict, "priors_type" str, and "exoticLD" dict. Used to perform linear least-squares fitting on SLCs to get estimate of depth and errors.
    :param MCMCfit_spec: dict of "do" bool, "exoplanet_params" dict, "systematics" tuple of float, "limb_darkening_model" dict, "fixed_param" dict, "priors_dict" dict, "priors_type" str, "exoticLD" dict, "N_walkers" int, "N_steps" int, and "est_err" float. Used to perform MCMC fitting on WLC to get depth.
    :param save_plots: dict of bools "do", "fit_and_residuals", "posteriors", "corner", "chains", and "spectrum". Whether to save plots and if so, which plots to save.
    :param reference_spectra: dict of lists "reference_wavelengths", "reference_depths", "reference_errs", "reference_names". Spectra from other papers that you want to plot for comparison.
    :return: fitted light curves and plots saved to subdirectories in the given outdir.
'''
    t0 = time.time()
    print("Beginning Stage 5 analysis...")
    if event_type == "Transit":
        event_type = "transit"
    elif event_type == "Eclipse":
        event_type = "eclipse"

    while event_type not in ("transit", "eclipse"):
        print("Juniper Stage 5 does recognize the supplied event_type ", event_type, " and cannot proceed.")
        event_type = str(input('Allowed event_type inputs are "transit" and "eclipse". Specify event type:'))
    
    plotdir = os.path.join(outdir, "plots")
    textdir = os.path.join(outdir, "fits")
    specdir = os.path.join(outdir, "spectra")
    
    # First, we need to get the paths to the WLC and SLCs in the filesdir.
    WLC_path = os.path.join(filesdir, "wlc.txt")
    WLC_wavs_path = os.path.join(filesdir, "wlc_wvs.txt")
    SLC_all_paths = sorted(glob.glob(os.path.join(filesdir, "slc*")))
    SLC_paths = [i for i in SLC_all_paths if "wvs" not in i]
    SLC_wavs_paths = [i for i in SLC_all_paths if "wvs" in i]
    
    if (LSQfit_WLC["do"] or MCMCfit_WLC["do"]):
        # Get the arrays needed to do fitting.
        print("Loading data needed for WLC fitting...")
        wlc, timestamps = read_light_curve(WLC_path)
        wavmin, wavmax, central_lam = read_wvs(WLC_wavs_path)
        print("Data loaded.")
    
    if LSQfit_WLC["do"]:
        print("Performing linear least-squares fitting of WLC to extract revised estimates of system parameters...")
        exoticLD = LSQfit_WLC["exoticLD"]
        exoticLD["spectral_range"] = (wavmin*10**4, wavmax*10**4) # need to update the spectral_range key, in AA.
        fit_theta, fit_corr_bat_lc, interp_t, residuals, err = LSQfit(timestamps, wlc, event_type,
                                                                      exoplanet_params=LSQfit_WLC["exoplanet_params"],
                                                                      systematics=LSQfit_WLC["systematics"],
                                                                      limb_darkening_model=LSQfit_WLC["limb_darkening_model"],
                                                                      fixed_param=LSQfit_WLC["fixed_param"],
                                                                      priors_dict=LSQfit_WLC["priors_dict"],
                                                                      priors_type=LSQfit_WLC["priors_type"],
                                                                      exoticLD=exoticLD,
                                                                      reject_threshold=reject_threshold)
            
        if (save_plots["do"] and save_plots["fit_and_residuals"]):
            if not os.path.exists(plotdir):
                os.makedirs(plotdir)
            fig, axes = utils.plot_fit_and_res(timestamps, wlc, err=np.array([0 for i in timestamps]),
                                                model=fit_corr_bat_lc, interp_t=interp_t, residuals=residuals)
            plt.savefig(os.path.join(plotdir, "LSQfit_fitres_WLC.pdf"), bbox_inches="tight")
            plt.close()
        # Write results of fit to .txt file.
        dummy_dict = {}
        dummy_dict["rp"] = 0
        dummy_dict["fp"] = 0
        utils.write_run_to_text(outdir=textdir,
                                outfile="LSQfit_fit_WLC.txt",
                                fit_theta=fit_theta,
                                err_theta=dummy_dict,
                                fit_err=err,
                                exoplanet_params=LSQfit_WLC["exoplanet_params"],
                                limb_darkening_model=LSQfit_WLC["limb_darkening_model"],
                                fixed_param=LSQfit_WLC["fixed_param"],
                                priors_dict=LSQfit_WLC["priors_dict"],
                                priors_type="uniform",
                                N_walkers="NA",
                                N_steps="NA",
                                exoticLD=exoticLD)
            
        # Update exoplanet_params based on LSQfit information.
        parameters_to_update = fit_theta.keys()
        for key in parameters_to_update:
            if key in ("a1","a2"):
                pass
            else:
                MCMCfit_WLC["exoplanet_params"][key] = fit_theta[key]
        MCMCfit_WLC["systematics"] = (fit_theta["a1"],fit_theta["a2"])
    
    if MCMCfit_WLC["do"]:
        print("Performing Markov Chain Monte Carlo fitting of WLC to extract system parameters with estimated uncertainties...")
        exoticLD = MCMCfit_WLC["exoticLD"]
        exoticLD["spectral_range"] = (wavmin*10**4, wavmax*10**4) # need to update the spectral_range key, in AA.
        if LSQfit_WLC["do"]:
            wlc, timestamps, n_reject = reject_outliers(timestamps, wlc, residuals, sigma=reject_threshold, raise_alarm=raise_alarm)
        try:
            wlc_err = np.array([err for i in timestamps])
        except:
            wlc_err = np.array([MCMCfit_WLC["est_err"] for i in timestamps])
        theta, theta_err, err, plotting_items = MCMCfit(timestamps, wlc, wlc_err, event_type,
                                                        exoplanet_params=MCMCfit_WLC["exoplanet_params"],
                                                        systematics=MCMCfit_WLC["systematics"],
                                                        limb_darkening_model=MCMCfit_WLC["limb_darkening_model"],
                                                        fixed_param=MCMCfit_WLC["fixed_param"],
                                                        priors_dict=MCMCfit_WLC["priors_dict"],
                                                        exoticLD=exoticLD,
                                                        priors_type=MCMCfit_WLC["priors_type"],
                                                        N_walkers=MCMCfit_WLC["N_walkers"],
                                                        N_steps=MCMCfit_WLC["N_steps"])
        if save_plots["do"]:
            if not os.path.exists(plotdir):
                os.makedirs(plotdir)
            if save_plots["fit_and_residuals"]:
                fig, axes = utils.plot_fit_and_res(timestamps, wlc, wlc_err,
                                                   model=plotting_items[5], interp_t=plotting_items[7], residuals=plotting_items[6])
                plt.savefig(os.path.join(plotdir, "MCMCfit_fitres_WLC.pdf"), bbox_inches="tight")
                plt.close()
            if save_plots["posteriors"]:
                fig, axes = utils.plot_post(ndim=plotting_items[0],
                                            samples=plotting_items[1],
                                            labels=plotting_items[3],
                                            n=plotting_items[4])
                plt.savefig(os.path.join(plotdir, "MCMCfit_post_WLC.pdf"), bbox_inches="tight")
                plt.close()
            if save_plots["corner"]:
                fig = utils.plot_corner(flat_samples=plotting_items[2],
                                        labels=plotting_items[3])
                plt.savefig(os.path.join(plotdir, "MCMCfit_corner_WLC.pdf"), bbox_inches="tight")
                plt.close()
            if save_plots["chains"]:
                fig, axes = utils.plot_chains(ndim=plotting_items[0],
                                              samples=plotting_items[1],
                                              labels=plotting_items[3])
                plt.savefig(os.path.join(plotdir, "MCMCfit_chains_WLC.pdf"), bbox_inches="tight")
                plt.close()

        # Write results of fit to .txt file.
        utils.write_run_to_text(outdir=textdir,
                                outfile="MCMCfit_fit_WLC.txt",
                                fit_theta=theta,
                                err_theta=theta_err,
                                fit_err=err,
                                exoplanet_params=MCMCfit_WLC["exoplanet_params"],
                                limb_darkening_model=MCMCfit_WLC["limb_darkening_model"],
                                fixed_param=MCMCfit_WLC["fixed_param"],
                                priors_dict=MCMCfit_WLC["priors_dict"],
                                priors_type=MCMCfit_WLC["priors_type"],
                                N_walkers=MCMCfit_WLC["N_walkers"],
                                N_steps=MCMCfit_WLC["N_steps"],
                                exoticLD=exoticLD)
        
        # Update exoplanet_params based on MCMCfit information.
        parameters_to_update = theta.keys()
        for key in parameters_to_update:
            if key in ("a1","a2"):
                pass
            else:
                LSQfit_spec["exoplanet_params"][key] = theta[key]
                MCMCfit_spec["exoplanet_params"][key] = theta[key]
        LSQfit_spec["systematics"] = (theta["a1"],theta["a2"])
        MCMCfit_spec["systematics"] = (theta["a1"],theta["a2"])
    
    if (LSQfit_spec["do"] or MCMCfit_spec["do"]):
        # Get the arrays needed to do fitting.
        print("Loading data needed for SLC fitting...")
        slc = []
        timestamps = []
        for path in SLC_paths:
            lc, t = read_light_curve(path)
            slc.append(lc)
            timestamps.append(t)
        #slc = np.array(slc)
        #timestamps = np.array(timestamps)
        wavelengths = []
        central_wavelengths = []
        for path in SLC_wavs_paths:
            wavmin, wavmax, central_lam = read_wvs(path)
            wavelengths.append((wavmin, wavmax))
            central_wavelengths.append(central_lam)
        print("Data loaded.")

    if LSQfit_spec["do"]:
        print("Performing linear least-squares fitting of SLCs to extract error estimates for use in MCMC fitting...")
        LD1 = []
        LD2 = []
        LSQ_thetas = {}
        LSQ_errs = {}
        LSQ_res = {}
        for l, wavelength in enumerate(central_wavelengths):
            # Iterate by wavelength.
            print("Operating on SLC wavelength {} micron.".format(wavelength))
            exoticLD = LSQfit_spec["exoticLD"]
            exoticLD["spectral_range"] = (wavelengths[l][0]*10**4, wavelengths[l][1]*10**4) # need to update the spectral_range key, in AA.
            try:
                fit_theta, fit_corr_bat_lc, interp_t, residuals, err = LSQfit(timestamps[l], slc[l], event_type,
                                                                            exoplanet_params=LSQfit_spec["exoplanet_params"],
                                                                            systematics=LSQfit_spec["systematics"],
                                                                            limb_darkening_model=LSQfit_spec["limb_darkening_model"],
                                                                            fixed_param=LSQfit_spec["fixed_param"],
                                                                            priors_dict=LSQfit_spec["priors_dict"],
                                                                            priors_type=LSQfit_spec["priors_type"],
                                                                            exoticLD=exoticLD,
                                                                            reject_threshold=reject_threshold)
            except ZeroDivisionError:
                fit_theta = None
            if fit_theta is None:
                print("Skipped fit for {}.".format(wavelength))
            else:
                print("Finished fit for {}. Saving outputs...".format(str(wavelength)))
                LD = LSQfit_spec["exoplanet_params"]["LD_coeffs"]
                LD1.append(LD[0])
                LD2.append(LD[1])
                LSQ_thetas[str(wavelength)] = fit_theta
                LSQ_errs[str(wavelength)] = err
                LSQ_res[str(wavelength)] = residuals
                if save_plots["do"]:
                    if not os.path.exists(plotdir):
                        os.makedirs(plotdir)
                    if save_plots["fit_and_residuals"]:
                        fig, axes = utils.plot_fit_and_res(timestamps[l], slc[l], err=np.array([0 for i in timestamps[l]]),
                                                           model=fit_corr_bat_lc, interp_t=interp_t, residuals=residuals)
                        plt.savefig(os.path.join(plotdir, "LSQfit_fitres_SLC_%.3fmu.pdf" % wavelength), bbox_inches="tight")
                        plt.close()
                # Write results of fit to .txt file.
                dummy_dict = {}
                dummy_dict["rp"] = 0
                dummy_dict["fp"] = 0
                utils.write_run_to_text(outdir=textdir,
                                        outfile="LSQfit_fit_SLC_%.3fmu.txt" % wavelength,
                                        fit_theta=fit_theta,
                                        err_theta=dummy_dict,
                                        fit_err=err,
                                        exoplanet_params=LSQfit_spec["exoplanet_params"],
                                        limb_darkening_model=LSQfit_spec["limb_darkening_model"],
                                        fixed_param=LSQfit_spec["fixed_param"],
                                        priors_dict=LSQfit_spec["priors_dict"],
                                        priors_type="uniform",
                                        N_walkers="NA",
                                        N_steps="NA",
                                        exoticLD=exoticLD)
    #with open("quadratic_LDs.txt",mode='w') as f:
    #    for i, j in zip(LD1, LD2):
    #        f.write("{},{}\n".format(i,j))
    #print(1/0)
    if MCMCfit_spec["do"]:
        print("Performing Markov Chain Monte Carlo fitting of SLCs to extract transit spectrum...")
        if not os.path.exists(specdir):
            os.makedirs(specdir)
        MCMC_thetas = {}
        MCMC_thetaerrs = {}
        MCMC_errs = {}
        MCMC_SDNRs = {}
        for l, wavelength in enumerate(central_wavelengths):
            # Iterate by wavelength.
            print("Operating on SLC wavelength {} micron.".format(wavelength))
            exoticLD = MCMCfit_spec["exoticLD"]
            exoticLD["spectral_range"] = (wavelengths[l][0]*10**4, wavelengths[l][1]*10**4) # need to update the spectral_range key, in AA.
            if LSQfit_spec["do"]:
                try:
                    slc[l], timestamps[l], n_reject = reject_outliers(timestamps[l], slc[l], LSQ_res[str(wavelength)], sigma=reject_threshold, raise_alarm=raise_alarm)
                    do_fit = True
                except KeyError:
                    print("Wavelength {} micron was not fitted by LSQ methods, skipping...".format(wavelength))
                    do_fit = False
                    theta = None
            else:
                do_fit = True
            if do_fit:
                try:
                    print("LSQ error: ", LSQ_errs[str(wavelength)])
                    slc_err = np.array([LSQ_errs[str(wavelength)] for i in timestamps[l]])
                except:
                    slc_err = np.array([MCMCfit_spec["est_err"] for i in timestamps[l]])
                try:
                    theta, theta_err, err, plotting_items = MCMCfit(timestamps[l], slc[l], slc_err, event_type,
                                                                    exoplanet_params=MCMCfit_spec["exoplanet_params"],
                                                                    systematics=MCMCfit_spec["systematics"],
                                                                    limb_darkening_model=MCMCfit_spec["limb_darkening_model"],
                                                                    fixed_param=MCMCfit_spec["fixed_param"],
                                                                    priors_dict=MCMCfit_spec["priors_dict"],
                                                                    exoticLD=exoticLD,
                                                                    priors_type=MCMCfit_spec["priors_type"],
                                                                    N_walkers=MCMCfit_spec["N_walkers"],
                                                                    N_steps=MCMCfit_spec["N_steps"])
                except ZeroDivisionError:
                    theta = None
            if theta is None:
                print("Skipped fit for {}.".format(wavelength))
            else:
                print("Finished fit for {}. Saving outputs...".format(str(wavelength)))
                MCMC_thetas[str(wavelength)] = theta
                MCMC_thetaerrs[str(wavelength)] = theta_err
                MCMC_errs[str(wavelength)] = err
                MCMC_SDNRs[str(wavelength)] = np.std(plotting_items[6])*10**6
                if save_plots["do"]:
                    if not os.path.exists(plotdir):
                        os.makedirs(plotdir)
                    if save_plots["fit_and_residuals"]:
                        fig, axes = utils.plot_fit_and_res(timestamps[l], slc[l], slc_err,
                                                            model=plotting_items[5], interp_t=plotting_items[7], residuals=plotting_items[6])
                        plt.savefig(os.path.join(plotdir, "MCMCfit_fitres_SLC_%.3fmu.pdf" % wavelength), bbox_inches="tight")
                        plt.close()
                    if save_plots["posteriors"]:
                        fig, axes = utils.plot_post(ndim=plotting_items[0],
                                                    samples=plotting_items[1],
                                                    labels=plotting_items[3],
                                                    n=plotting_items[4])
                        plt.savefig(os.path.join(plotdir, "MCMCfit_post_SLC_%.3fmu.pdf" % wavelength), bbox_inches="tight")
                        plt.close()
                    if save_plots["corner"]:
                        fig = utils.plot_corner(flat_samples=plotting_items[2],
                                                labels=plotting_items[3])
                        plt.savefig(os.path.join(plotdir, "MCMCfit_corner_SLC_%.3fmu.pdf" % wavelength), bbox_inches="tight")
                        plt.close()
                    if save_plots["chains"]:
                        fig, axes = utils.plot_chains(ndim=plotting_items[0],
                                                      samples=plotting_items[1],
                                                      labels=plotting_items[3])
                        plt.savefig(os.path.join(plotdir, "MCMCfit_chains_SLC_%.3fmu.pdf" % wavelength), bbox_inches="tight")
                        plt.close()
                
                # Write results of fit to .txt file.
                utils.write_run_to_text(outdir=textdir,
                                        outfile="MCMCfit_fit_SLC_%.3fmu.txt" % wavelength,
                                        fit_theta=theta,
                                        err_theta=theta_err,
                                        fit_err=err,
                                        exoplanet_params=MCMCfit_spec["exoplanet_params"],
                                        limb_darkening_model=MCMCfit_spec["limb_darkening_model"],
                                        fixed_param=MCMCfit_spec["fixed_param"],
                                        priors_dict=MCMCfit_spec["priors_dict"],
                                        priors_type=MCMCfit_spec["priors_type"],
                                        N_walkers=MCMCfit_spec["N_walkers"],
                                        N_steps=MCMCfit_spec["N_steps"],
                                        exoticLD=exoticLD)
        
        for depth_type in ("rprs","rprs2","aoverlap","fpfs"):
            if (depth_type == "fpfs" and event_type != "eclipse"):
                pass
            elif (depth_type != "fpfs" and event_type == "eclipse"):
                pass
            else:
                # Write spectrum out.
                wavelengths, depths, depth_errs = utils.get_spectrum(MCMC_thetas, MCMC_thetaerrs, MCMC_SDNRs,
                                                                            exoplanet_params=MCMCfit_spec["exoplanet_params"],
                                                                            depth_type=depth_type, ignore_high_SDNR=False)
                outfile = "MCMC_spectrum_{}.txt".format(depth_type)
                utils.write_spectrum(specdir, outfile, wavelengths, depths, depth_errs)
                if save_plots["do"]:
                    if not os.path.exists(plotdir):
                        os.makedirs(plotdir)
                    if save_plots["spectrum"]:
                        if depth_type != "fpfs":
                            fig, axes = utils.plot_transit_spectrum(wavelengths, depths, depth_errs,
                                                                    reference_wavelengths=reference_spectra["reference_wavelengths"],
                                                                    reference_depths=reference_spectra["reference_depths"],
                                                                    reference_errs=reference_spectra["reference_errs"],
                                                                    reference_names=reference_spectra["reference_names"],
                                                                    ylim=(0.95*min(depths),1.05*max(depths)))
                            plt.savefig(os.path.join(plotdir, "transit_spectrum_{}.pdf".format(depth_type)), bbox_inches="tight")
                            plt.close()
                        else:
                            fig, axes = utils.plot_eclipse_spectrum(wavelengths, depths, depth_errs,
                                                                    reference_wavelengths=reference_spectra["reference_wavelengths"],
                                                                    reference_depths=reference_spectra["reference_depths"],
                                                                    reference_errs=reference_spectra["reference_errs"],
                                                                    reference_names=reference_spectra["reference_names"],
                                                                    ylim=(0.95*min(depths),1.05*max(depths)))
                            plt.savefig(os.path.join(plotdir, "eclipse_spectrum_{}.pdf".format(depth_type)), bbox_inches="tight")
                            plt.close()
                    fig, axes = utils.plot_SDNRs(SDNRs=MCMC_SDNRs)
                    plt.savefig(os.path.join(plotdir, "{}_fit_residuals.pdf".format(event_type)), bbox_inches="tight")
                    plt.close()
                # Compute final transit spectrum with chosen depths ignored.
                for limit in []:
                    print("On SDNR limit {}...".format(limit))
                    wavelengths, depths, depth_errs = utils.get_spectrum(MCMC_thetas, MCMC_thetaerrs, MCMC_SDNRs,
                                                                                exoplanet_params=MCMCfit_spec["exoplanet_params"],
                                                                                depth_type=depth_type, ignore_high_SDNR=True,
                                                                                SDNRlimit=limit)
                    try:
                        outfile = "MCMC_spectrum{}ppm_{}.txt".format(limit, depth_type)
                        utils.write_spectrum(specdir, outfile, wavelengths, depths, depth_errs)

                        print("Depth spectrum:")
                        print([float(x) for x in depths])
                        if save_plots["do"]:
                            if not os.path.exists(plotdir):
                                os.makedirs(plotdir)
                            if save_plots["spectrum"]:
                                if depth_type != "fpfs":
                                    fig, axes = utils.plot_transit_spectrum(wavelengths, depths, depth_errs,
                                                                            reference_wavelengths=reference_spectra["reference_wavelengths"],
                                                                            reference_depths=reference_spectra["reference_depths"],
                                                                            reference_errs=reference_spectra["reference_errs"],
                                                                            reference_names=reference_spectra["reference_names"],
                                                                            ylim=(0.95*min(depths),1.05*max(depths)))
                                    plt.savefig(os.path.join(plotdir, "transit_spectrum{}ppm_{}.pdf".format(limit, depth_type)), bbox_inches="tight")
                                    plt.close()
                                else:
                                    fig, axes = utils.plot_eclipse_spectrum(wavelengths, depths, depth_errs,
                                                                            reference_wavelengths=reference_spectra["reference_wavelengths"],
                                                                            reference_depths=reference_spectra["reference_depths"],
                                                                            reference_errs=reference_spectra["reference_errs"],
                                                                            reference_names=reference_spectra["reference_names"],
                                                                            ylim=(0.50*min(depths),1.50*max(depths)))
                                    plt.savefig(os.path.join(plotdir, "eclipse_spectrum{}ppm_{}.pdf".format(limit, depth_type)), bbox_inches="tight")
                                    plt.close()
                    except ZeroDivisionError:
                        print("Spectrum cannot be plotted as it has nans!")
                        try:
                            plt.close()
                        except:
                            pass
    
    tf = time.time() - t0
    print("Stage 5 analysis resolved in %.3f seconds = %.3f minutes." % (tf, tf/60))

def read_light_curve(filepath):
    '''
    Reads out the light curve .txt located at filepath.
    
    :param filepath: str. Where the light curve .txt object is located.
    :return: lc_n, t object.
    '''
    lc = []
    t = []
    with open(filepath) as f:
            line = f.readline() 
            while line[0] == '#':
                # Read past comments.
                line = f.readline()
            while line != '':
                line = str.split(line)#, sep='   ')
                
                # Extract useful info.
                time = float(line[0]) # time in days relative to mid-transit or mid-eclipse
                flux = float(str.replace(line[1],'\n','')) # normalized flux
                
                t.append(time)
                lc.append(flux)
                
                line = f.readline()
    return np.array(lc), np.array(t)

def read_wvs(filepath):
    '''
    Reads out the wavelengths .txt located at filepath.
    
    :param filepath: str. Where the wavelengths .txt object is located.
    :return: wavmin, wavmax, central_lam.
    '''
    items = []
    with open(filepath) as f:
            line = f.readline() 
            while line[0] == '#':
                # Read past comments.
                line = f.readline()
            while line != '':
                line = str.split(line)#, sep='   ')
                
                # Extract useful info.
                item = str.replace(line[0],'\n','')
                item = str.replace(item,' ','')
                item = float(item)
                
                items.append(item)
                
                line = f.readline()
    wavmin, wavmax, central_lam = items
    return wavmin, wavmax, central_lam

def LSQfit(t, lc, event_type, exoplanet_params, systematics, limb_darkening_model, fixed_param, priors_dict, priors_type, exoticLD, reject_threshold=None):
    '''
    Performs linear least-squares fit of transit model to provided light curve.

    :param t: 1D array. Timestamps for each flux point in the array.
    :param lc: 1D array. A light curve to fit a model to.
    :param event_type: str. "transit" or "eclipse".
    :param exoplanet_params: dict of float. Contains keywords "t0", "period", "rp", "aoR", "inc", "ecc", "lop".
    :param systematics: tuple of float. Contains parameters (a1, a2) for a linear-in-time fit sys(t) = a1*t+a2.
    :param limb_darkening_model: dict. Contains "model_type" str which defines model choice (e.g. quadratic, 4-param), "stellar_params" tuple of (M_H, Teff, logg) or None if not using, "coefficients" keyword containing tuple of floats which can be fixed or fitted for (in the latter case, "coefficients" defines the starting guess). If LD is supplied by exotic, "coefficients" is ignored.
    :param fixed_param. dict of bools. Keywords are parameters that can be held fixed or opened for fitting. If True, parameter will be held fixed. If False, parameter is allowed to be fitted.
    :param priors_dict: dict of tup of float. Keywords are parameters. Defines the min and max value that LSQfit can return for the given parameter.
    :param priors_type: str. Choices are "uniform" or "gaussian". Sets how to interpret priors_dict.
    :param exoticLD: dict. Contains "available" bool for whether EXoTiC-LD is on this system, "ld_data_path" str of where the exotic_ld_data directory is located, "spectral_range" wavelength range being covered.
    :param reject_threshold: float or None. Sigma at which to reject outliers from the linear least-squares residuals when performing MCMC fitting.
    :return: theta dict of all fitted params, 1D array of fitted model, 1D array of residuals, and float of SDNR in ppm.
    '''
    # First, confirm that the light curve is not a failed fit.
    if all(1 == x for x in lc):
        print("Given light curve is a failed fit, returning None...")
        return None, None, None, None, None

    # Then start unpacking systematics and stellar_params.
    a1, a2 = systematics
    exoplanet_params["model_type"], stellar_params, exoplanet_params["LD_coeffs"] = (limb_darkening_model["model_type"],
                                                                                     limb_darkening_model["stellar_params"],
                                                                                     limb_darkening_model["initial_guess"])
    
    if (exoticLD["available"]):
        exoplanet_params = utils.get_exotic_coefficients(exoplanet_params, stellar_params, exoticLD)
        
    # Special condition for those using Kipping2013.
    if (exoplanet_params["model_type"] == "kipping2013"):
        using_kipping = True
        exoplanet_params["model_type"] = "quadratic"
    else:
        using_kipping = False
    
    print("Using ld coeffs initial guess: ", exoplanet_params["LD_coeffs"])
    residuals, trimmed_residuals, theta_guess, fit_theta, fit_corr_bat_lc, interp_t = lsqfit(event_type, exoplanet_params, fixed_param, priors_dict,
                                                                                             a1, a2, t, lc, using_kipping, priors_type, reject_threshold)
    
    if trimmed_residuals is not None:
        err = np.std(trimmed_residuals)
        print("Standard deviation of the trimmed residuals: %.0f ppm" % (err*10**6))
    else:
        err = np.std(residuals)
        print("Standard deviation of the residuals: %.0f ppm" % (err*10**6))
    
    return fit_theta, fit_corr_bat_lc, interp_t, residuals, err

def lsqfit(event_type, exoplanet_params, fixed_param, priors_dict, a1, a2, t, lc, using_kipping, priors_type, reject_threshold=None):
    '''
    Invokes least-squares fitting and handles outputs for the LSQfit.do() routine.

    :param event_type: str. "transit" or "eclipse". Needed for initializing the correct batman model.
    :param exoplanet_params: dict of float. Contains keywords "t0", "period", "rp", "aoR", "inc", "ecc", "lop".
    :param fixed_param. dict of bools. Keywords are parameters that can be held fixed or opened for fitting. If True, parameter will be held fixed. If False, parameter is allowed to be fitted.
    :param priors_dict: dict of tup of float. Keywords are parameters. Defines the min and max value that LSQfit can return for the given parameter.
    :param a1: float. The slope of linear-in-time systematics model.
    :param a2: float. The intercept of linear-in-time systematics model.
    :param t: 1D array. Timestamps for each flux point in the array.
    :param lc: 1D array. A light curve to fit a model to.
    :param using_kipping: bool. Whether Kipping 2013 parameterization of LD coefficients is in use.
    :param reject_threshold: float or None. Sigma at which to reject outliers from the linear least-squares residuals when performing MCMC fitting.
    :return: 1D array of residuals to the best fit, dict of theta_guess initial guess, dict of fit_theta solution, 1D array fit_corr_bat_lc of the fitted model, 1D of the interpolated time for plotting purposes.
    '''
    # Initialize batman model.
    params = utils.initialize_batman_params(exoplanet_params, event_type)
    
    if event_type == "transit":
        init_bat_model = batman.TransitModel(params, t)
    elif event_type == "eclipse":
        init_bat_model = batman.TransitModel(params, t, transittype="secondary")
    
    # Initialize guess dictionary.
    theta_guess = utils.build_theta_dict(a1, a2, exoplanet_params, fixed_param)

    # Commence fitting using fit_model routine.
    fit_theta, fit_theta_arr, modified_keys, fit_corr_bat_lc = fit_model(theta_guess, init_bat_model,
                                                                         params, t, lc, priors_dict, using_kipping, priors_type)
    
    # Compute residuals of the best-fit model.
    residuals = residuals_(fit_theta_arr, fit_theta,
                           modified_keys, init_bat_model, params, t, lc, using_kipping)
    if reject_threshold is not None:
        trimmed_lc, trimmed_t, n_reject = reject_outliers(np.copy(t), np.copy(lc), residuals, sigma=reject_threshold, raise_alarm=10)
        if event_type == "transit":
            init_bat_model = batman.TransitModel(params, trimmed_t)
        elif event_type == "eclipse":
            init_bat_model = batman.TransitModel(params, trimmed_t, transittype="secondary")
        trimmed_residuals = residuals_(fit_theta_arr, fit_theta,
                                       modified_keys, init_bat_model, params, trimmed_t, trimmed_lc, using_kipping)
    else:
        trimmed_residuals = None
    
    # Compute a higher time resolution version of fit_corr_bat_lc for plotting puproses.
    interp_t = np.linspace(np.min(t),np.max(t),1000)
    if event_type == "transit":
        init_bat_model = batman.TransitModel(params, interp_t)
    elif event_type == "eclipse":
        init_bat_model = batman.TransitModel(params, interp_t, transittype="secondary")
    polyfit = fit_theta["a1"]*interp_t + fit_theta["a2"]
    fit_corr_bat_lc = init_bat_model.light_curve(params)*polyfit
    return residuals, trimmed_residuals, theta_guess, fit_theta, fit_corr_bat_lc, interp_t

def fit_model(theta_guess, init_bat_model, params, t, lc, priors_dict, using_kipping, priors_type):
    '''
    Fits a batman transit model to the supplied data using the supplied initial guess and priors.
    
    :param theta_guess: dict of float. Contains keywords based on which parameters are being fitted for.
    :param init_bat_model. batman model oobject. The initialized batman transit model which will be modified for fitting.
    :param params: batman params object. The parameters being used for the batman model, some of which are being fitted.
    :param t: 1D array. Timestamps for each flux point in the array.
    :param lc: 1D array. A light curve to fit a model to.
    :param priors_dict: dict of tup of float. Keywords are parameters. Defines the min and max value that LSQfit can return for the given parameter.
    :param using_kipping: bool. Whether Kipping 2013 parameterization of LD coefficients is in use.
    :return: fit_theta dict of the solution, fit_theta_arr 1D array of the prior, modified_keys lst of what parameters were fitted, fit_corr_bat_lc 1D array of the fitted model.
    '''
    original_theta_guess = deepcopy(theta_guess)
    # Need to unpack the limb darkening coefficients and build bounds object
    # Check limb dark type
    if (params.limb_dark == "quadratic" and using_kipping):
        print("Kipping2013 formulation found, switching priors appropriately...")
        ld_lower = 0
        ld_upper = 1
    
    else:
        ld_lower = -1
        ld_upper = 2
    
    theta_arr, modified_keys = utils.turn_dict_to_array(theta_guess)
    bounds = utils.make_LSQ_bounds_object(theta_guess, priors_dict, priors_type, ld_lower, ld_upper)    
    
    opt_result = least_squares(residuals_,
                               theta_arr,
                               bounds=bounds,
                               args=(theta_guess, modified_keys, init_bat_model, params, t, lc, using_kipping))
    fit_theta_arr = opt_result.x
    print("Fitted: ", fit_theta_arr)
    print("Least squares finished with status:", opt_result.status)
    print("Output message: ", opt_result.message)
    print("Success status: ", opt_result.success)
    # Return fit_theta to dictionary format, and unpack fitted LD coeffs back into list.
    fit_theta = utils.turn_array_to_dict(modified_keys, fit_theta_arr)
    
    rchi2 = (residuals_(fit_theta_arr, fit_theta, modified_keys, init_bat_model, params, t, lc, using_kipping)**2).sum()/(len(lc)-len(theta_guess))
    
    print('Guess', original_theta_guess)
    print('Fitted', fit_theta)
    
    fit_corr_bat_lc = modify_model(fit_theta, init_bat_model, params, t)
    res = residuals_(fit_theta_arr, fit_theta, modified_keys, init_bat_model, params, t, lc, using_kipping)
    
    chi2 = sum(res*res)
    print('Chi-square =', chi2)
    
    dof = len(lc)-len(theta_guess)
    print('Deg of freedom =', dof)
    
    print('Reduced Chi-square =', rchi2)
    return fit_theta, fit_theta_arr, modified_keys, fit_corr_bat_lc

def residuals_(theta_arr, theta, modified_keys, init_bat_model, params, t, lc, using_kipping):
    '''
    Modifies the model based on least-squares methods and returns the residuals of the modified model.

    :param theta_arr: 1D array. Theta (dict) unpacked into an array, needed for compatability with 
    :param theta: dict of float. Contains keywords based on which parameters are being fitted for.
    :param modified_keys: lst of str. Which keys were being fitted for, including LD coefficients.
    :param init_bat_model: the currently-initialized batman model that we are about to update the param on.
    :param params: batman params object. The parameters being used for the batman model, some of which are being fitted.
    :param t: 1D array. Timestamps for each flux point in the array.
    :param lc: 1D array. A light curve to fit a model to.
    :param using_kipping: bool. Whether Kipping 2013 parameterization of LD coefficients is in use.
    :return: residuals 1D array of the residuals of the fit.
    '''
    # theta-arr is the array which will be modified. theta is the dictionary to which these
    # these changes must be broadcast back to. Need to be delicate handling LD coeffs.
    theta = utils.broadcast_array_back_to_dict(theta_arr, theta, modified_keys, using_kipping)
    
    full_bat_model = modify_model(theta, init_bat_model, params, t)
    
    residuals = (lc-full_bat_model)
    
    return residuals

def modify_model(theta, init_bat_model, params, t):
    '''
    Modifies the current batman model using the new theta and returns the new modelled light curve.

    :param theta: dict of float. Parameters that are being fitted.
    :param init_bat_model: the currently-initialized batman model that we are about to update the param on.
    :param params: batman params object. The parameters being used for the batman model, some of which are being fitted.
    :param t: 1D array. Timestamps for each flux point in the array.
    :return: full_bat_model 1D array of the model light curve with linear systematic multiplied in.
    '''
    a1, a2, params = utils.update_params(theta, params)
    model_lc = init_bat_model.light_curve(params)
    polyfit = a1*t + a2
    
    full_bat_model = model_lc*polyfit
    
    return full_bat_model

def MCMCfit(t, lc, err, event_type, exoplanet_params, systematics, limb_darkening_model, fixed_param, priors_dict, exoticLD, priors_type="uniform", N_walkers = 32, N_steps = 5000):
    '''
    Performs Markov Chain Monte Carlo fit of transit model to provided light curve.

    :param t: 1D array. Timestamps for each flux point in the array.
    :param lc: 1D array. A light curve to fit a model to.
    :param err: 1D array. The uncertainty on each lc point.
    :param event_type: str. "transit" or "eclipse".
    :param exoplanet_params: dict of float. Contains keywords "t0", "period", "rp", "aoR", "inc", "ecc", "lop".
    :param systematics: tuple of float. Contains parameters (a1, a2) for a linear-in-time fit sys(t) = a1*t+a2.
    :param limb_darkening_model: dict. Contains "model_type" str which defines model choice (e.g. quadratic, 4-param), "stellar_params" tuple of (M_H, Teff, logg) or None if not using, "coefficients" keyword containing tuple of floats which can be fixed or fitted for (in the latter case, "coefficients" defines the starting guess). If LD is supplied by exotic, "coefficients" is ignored.
    :param fixed_param. dict of bools. Keywords are parameters that can be held fixed or opened for fitting. If True, parameter will be held fixed. If False, parameter is allowed to be fitted.
    :param priors_dict: dict of tup of float. Keywords are parameters. If priors are "uniform", sets upper and lower bounds. If priors are "gaussian", sets mean and standard deviation.
    :param exoticLD: dict. Contains "available" bool for whether EXoTiC-LD is on this system, "ld_data_path" str of where the exotic_ld_data directory is located, "spectral_range" wavelength range being covered.
    :param priors_type: str. Choices are "uniform" or "gaussian". Type of priors to use.
    :param N_walkers: int. Number of chains to run in parallel.
    :param N_steps: int. Number of steps each chain takes.
    :return: theta dict of all fitted params, theta_err dict of 1sigma uncertainties on fitted params, float of SDNR in ppm, and tuple of items used only in plotting.
    '''
    # First, confirm that the light curve is not a failed fit.
    if all(1 == x for x in lc):
        print("Given light curve is a failed fit, returning None...")
        return None, None, None, None
    
    # Then start unpacking systematics and stellar_params.
    a1, a2 = systematics
    exoplanet_params["model_type"], stellar_params, exoplanet_params["LD_coeffs"] = (limb_darkening_model["model_type"],
                                                                                     limb_darkening_model["stellar_params"],
                                                                                     limb_darkening_model["initial_guess"])
    
    if (exoticLD["available"]):
        exoplanet_params = utils.get_exotic_coefficients(exoplanet_params, stellar_params, exoticLD)
        
    # Special condition for those using Kipping2013.
    if (exoplanet_params["model_type"] == "kipping2013"):
        using_kipping = True
        exoplanet_params["model_type"] = "quadratic"
    else:
        using_kipping = False
    
    print("Using ld coeffs initial guess: ", exoplanet_params["LD_coeffs"])

    # Set up params for batman
    params = utils.initialize_batman_params(exoplanet_params, event_type)
    
    # Initialize model.
    if event_type == "transit":
        bat_model = batman.TransitModel(params, t)
    elif event_type == "eclipse":
        bat_model = batman.TransitModel(params, t, transittype="secondary")

    # Initialize the guess.
    theta_guess = utils.build_theta_dict(a1, a2, exoplanet_params, fixed_param)
        
    # Convert the guess into an array.
    theta_arr, modified_keys = utils.turn_dict_to_array(theta_guess)

    # Run MCMC
    pos = theta_arr + 1e-4 * np.random.randn(N_walkers, theta_arr.shape[0])
    nwalkers, ndim = pos.shape
    print("Fitting %.0f parameters to data..." % ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(bat_model, modified_keys, params, priors_dict, priors_type, using_kipping, t, lc, err),)
    sampler.run_mcmc(pos, N_steps, progress=False);
    
    # These are the important results we need to use.
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=int(0.2*N_steps), flat=True)
    n = np.shape(samples[:,:,0])[0]*np.shape(samples[:,:,0])[1] # [:,:,1]? or will [:,:,0] suffice and be more general?
    labels = [key for key in modified_keys]

    theta, theta_err = get_result_from_post(ndim, flat_samples)
    
    # Need to turn theta back into a dict.
    theta = utils.turn_array_to_dict(modified_keys, theta)
    theta_err = utils.turn_array_to_dict(modified_keys, theta_err)
    
    # Reinitialize model.
    params = utils.initialize_batman_params(exoplanet_params, event_type)
    
    # Replace default params with fitted params as applicable.
    a1, a2, params = utils.update_params(theta, params)
    
    if event_type == "transit":
        bat_model = batman.TransitModel(params, t)
    elif event_type == "eclipse":
        bat_model = batman.TransitModel(params, t, transittype="secondary")
    polyfit = a1*t + a2
    model = bat_model.light_curve(params)*polyfit

    residuals = lc - model

    # Compute a higher time resolution version of model for plotting puproses.
    interp_t = np.linspace(np.min(t),np.max(t),1000)
    if event_type == "transit":
        bat_model = batman.TransitModel(params, interp_t)
    elif event_type == "eclipse":
        bat_model = batman.TransitModel(params, interp_t, transittype="secondary")
    polyfit = a1*interp_t + a2
    model = bat_model.light_curve(params)*polyfit

    plotting_items = (ndim, samples, flat_samples, labels, n, model, residuals, interp_t)
    err = np.std(residuals)
    print("Standard deviation of the residuals: %.0f ppm" % (err*10**6))
    
    return theta, theta_err, err, plotting_items
    
def log_likelihood(theta, bat_model, modified_keys, params, x, y, yerr, using_kipping):
    theta_dict = utils.turn_array_to_dict(modified_keys, theta)
    try:
        LD_coeffs = theta_dict["LD_coeffs"]
        fitting_LD = True
    except:
        fitting_LD = False
    
    a1, a2, params = utils.update_params(theta_dict, params)
    
    if (using_kipping and fitting_LD):
        # Change to Kipping parameters
        q1, q2 = LD_coeffs
        u1 = 2*np.sqrt(q1)*q2
        u2 = np.sqrt(q1)*(1-2*q2)
        params.u = [u1, u2]
    
    polyfit = a1*x + a2
    
    full_model = bat_model.light_curve(params)*polyfit
    sigma2 = yerr**2
    return -0.5 * np.sum((y - full_model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta, modified_keys, params, priors_dict, priors_type, using_kipping):
    # theta is an array. Need to turn it back into a dictionary.
    theta_dict = {}
    prior_prob = 0 # 0 or inf for uniform, else from gaussian
    
    theta_dict = utils.turn_array_to_dict(modified_keys, theta)
    
    a1 = theta_dict["a1"]
    a2 = theta_dict["a2"]
    #if "rp" in list(theta_dict.keys()):
    #    params.rp = theta_dict["rp"]
    #if "fp" in list(theta_dict.keys()):
    #    params.fp = theta_dict["fp"]
    try:
        LD_coeffs = theta_dict["LD_coeffs"]
    except:
        # Not fitting for LD_coeffs.
        pass
    
    # Now need to check the posteriors for all of these.
    checks_on_posteriors = ["T"]
    if not -5 < a1 < 5:
        checks_on_posteriors.append("F")
        
    if not -5 < a2 < 5:
        checks_on_posteriors.append("F")
    
    if "rp" in theta_dict.keys():
        if not 0.001 < theta_dict["rp"] < 100:
            checks_on_posteriors.append("F")
    # We must let even unphysical fp in, though it pains me as an astrophysicist to do so.
    #if "fp" in theta_dict.keys():
    #    if not 0 <= theta_dict["fp"] < 100:
    #        checks_on_posteriors.append("F")
    
    if using_kipping:
        umin, umax = (0, 1)
    elif (params.limb_dark == "quadratic" and not using_kipping):
        umin, umax = (-1, 2)
    else:
        umin, umax = (-5, 5)
    
    if "LD_coeff" in modified_keys:
        for u in LD_coeffs:
            if not umin <= u <= umax:
                checks_on_posteriors.append("F")

    for key in ("ecc","period","inc","lop","aoR","t0","t_secondary"):
        if key in modified_keys:
            if priors_type == "uniform":
                if not priors_dict[key][0] <= theta_dict[key] <= priors_dict[key][1]:
                    checks_on_posteriors.append("F")
            if priors_type == "gaussian":
                gauss_mu = priors_dict[key][0]
                gauss_sig = priors_dict[key][1]
                prior_prob += np.log(1.0/(np.sqrt(2*np.pi)*gauss_sig))-0.5*(theta_dict[key]-gauss_mu)**2/gauss_sig**2
    if isnan(prior_prob):
        return -np.inf
    if "F" not in checks_on_posteriors:
        return prior_prob
    return -np.inf

def log_probability(theta, bat_model, modified_keys, params, priors_dict, priors_type, using_kipping, x, y, yerr):
    lp = log_prior(theta, modified_keys, params, priors_dict, priors_type, using_kipping)
    if isnan(lp):
        return -np.inf
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, bat_model, modified_keys, params, x, y, yerr, using_kipping)

def get_result_from_post(ndim, flat_samples):
    theta = []
    theta_err = []
    for i in range(ndim):
        theta.append(np.percentile(flat_samples[:, i], 50))
        theta_err.append(np.std(flat_samples[:, i]))
    return np.array(theta), np.array(theta_err)

def reject_outliers(timestamps, lc, residuals, sigma, raise_alarm):
    print("Iterating twice to delete outliers...")
    n_del = 0
    del_times = []
    for i in range(2):
        res_mean = np.mean(residuals)
        res_sig = np.std(residuals)
        outliers = np.where(np.abs(residuals-res_mean) > sigma*res_sig)[0]
        n_del += len(outliers)
        for t in timestamps[outliers]:
            del_times.append(t)
        lc = np.delete(lc, outliers)
        timestamps = np.delete(timestamps, outliers)
        residuals = np.delete(residuals, outliers)
    print("{} outliers were deleted at these times: ".format(len(del_times)), del_times)
    if len(del_times) >= raise_alarm:
        print("alarm!")
    return lc, timestamps, len(del_times)