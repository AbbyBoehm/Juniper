import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer


def make_gif(oneD_spec, wav_sols, timestamps, inpt_dict):
    """Plots a gif of the extracted spectra over time.

    Args:
        oneD_spec (_type_): _description_
        wav_sols (_type_): _description_
        timestamps (_type_): _description_
        inpt_dict (_type_): _description_
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Creating gif of extracted spectra...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # create animation
    fig,ax = plt.subplots(figsize = (10,5))

    # plot first spectrum, get things started
    spec_line = ax.plot(wav_sols[0,:],oneD_spec[0,:],color='navy')
    ax.set_title('T = {:.5F} d'.format(timestamps[0]))
    ax.set_xlabel('wavelength [um]')
    ax.set_ylabel('flux [a.u.]')
    ax.set_xlim(np.nanmin(wav_sols),np.nanmax(wav_sols))
    ax.set_ylim(0,min(10*np.nanmean(oneD_spec),np.max(oneD_spec)))

    # initialize 
    def init():
        spec_line[0].set_data(wav_sols[0,:],oneD_spec[0,:])
        ax.set_title('T = {:.5F} d'.format(timestamps[0]))
        
        return spec_line

    # define animation function
    def animation_func(i,timestamps):
        # update line data
        spec_line[0].set_data(wav_sols[i,:],oneD_spec[i,:])
        ax.set_title('T = {:.5F} d'.format(timestamps[i]))
        
        return spec_line
        
    # create and plot animation
    animation = FuncAnimation(fig, animation_func, init_func = init, frames = np.shape(oneD_spec)[0], interval = 10,
                              fargs=(timestamps,))
    plt.tight_layout()

    # save animation
    if save_step:
        animation.save(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_spectrum.gif'),
                       writer = 'ffmpeg', fps = 120)

    if plot_step:
        plt.show(block = True)

    plt.close() # save memory

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)


def make_stack(oneD_spec, wav_sols, timestamps, inpt_dict):
    """Plots a stack of the extracted spectra over time.

    Args:
        oneD_spec (_type_): _description_
        wav_sols (_type_): _description_
        timestamps (_type_): _description_
        inpt_dict (_type_): _description_
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Creating stack of extracted spectra...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # create plot
    fig,ax = plt.subplots(figsize = (10,5))

    # plot all spectra on top of each other
    for i in range(oneD_spec.shape[0]):
        ax.plot(wav_sols[i,:],oneD_spec[i,:])
    ax.set_title('All 1D spectra')
    ax.set_xlabel(r'Wavelength [$\mu$m]')
    ax.set_ylabel('Flux [a.u.]')
    ax.set_xlim(np.nanmin(wav_sols),np.nanmax(wav_sols))
    ax.set_ylim(0,min(10*np.nanmean(oneD_spec),np.max(oneD_spec)))

    plt.tight_layout()

    # save plot
    if save_step:
        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_spectra.png'),
                    dpi=300,bbox_inches='tight')

    if plot_step:
        plt.show(block = True)

    plt.close() # save memory

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

def make_err_median(oneD_spec, wav_sols, oneD_err, inpt_dict):
    """Plots a stack of the median spectrum with error bars.

    Args:
        oneD_spec (_type_): _description_
        wav_sols (_type_): _description_
        oneD_err (_type_): _description_
        inpt_dict (_type_): _description_
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Creating median spectrum with error bars...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # create plot
    fig,ax = plt.subplots(figsize = (10,5))

    # get medians
    medwav = np.median(wav_sols,axis=0)
    medspec = np.median(oneD_spec,axis=0)
    mederr = np.median(oneD_err,axis=0)

    # plot median spectrum with error bars
    ax.plot(medwav,medspec,color='k')
    ax.errorbar(medwav,medspec,yerr=mederr,color='k',ls='none',
                capsize=3,marker='o',markersize=1)
    ax.set_title('Median 1D spectra with error bars')
    ax.set_xlabel(r'Wavelength [$\mu$m]')
    ax.set_ylabel('Flux [a.u.]')
    ax.set_xlim(np.nanmin(wav_sols),np.nanmax(wav_sols))
    ax.set_ylim(0,min(10*np.nanmean(oneD_spec),np.max(oneD_spec)))

    plt.tight_layout()

    # save plot
    if save_step:
        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_medspec-errs.png'),
                    dpi=300,bbox_inches='tight')

    if plot_step:
        plt.show(block = True)

    plt.close() # save memory

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)


def make_wlc(oneD_spec, wav_sols, timestamps, inpt_dict):
    """Plots the summed 1D spectrum at each time stamp.

    Args:
        oneD_spec (_type_): _description_
        wav_sols (_type_): _description_
        timestamps (_type_): _description_
        inpt_dict (_type_): _description_
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Creating stack of extracted spectra...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # create plot
    fig,ax = plt.subplots(figsize = (7,5))

    # sum each spectrum and plot
    wlc = []
    for i in range(oneD_spec.shape[0]):
        wlc.append(np.sum(oneD_spec[i,:]))
    ax.scatter(timestamps,wlc,color='k',marker='o')
    ax.set_title('Broad-band light curve')
    ax.set_xlabel('Exposure Time [BJD TDB]')
    ax.set_ylabel('Flux [a.u.]')

    plt.tight_layout()

    # save animation
    if save_step:
        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_broadband-final.png'),
                    dpi=300,bbox_inches='tight')

    if plot_step:
        plt.show(block = True)

    plt.close() # save memory

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
