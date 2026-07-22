import numpy as np

import corner
import matplotlib.pyplot as plt
from matplotlib.scale import get_scale_names
from mc3.stats import time_avg


def aspect_handler(img):
    """Handler for getting the right image and colorbar aspect ratios.

    Args:
        array (np.array): Image you want to plot.
    
    Returns:
        float or str, int: the correct aspect values for spectroscopic vs photometric plots.
    """
    # Check if img is mostly square.
    if abs((img.shape[0]/img.shape[1])-1)<1:
        return 1, 5
    else:
        return 'auto', 40

def img(array, aspect=1, title=None, vmin=None, vmax=None, norm=None, verbose=2):
    """Image plotting utility to plot the given 2D array.

    Args:
        array (np.array): Image you want to plot.
        aspect (str or int, optional): Aspect ratio. Useful for visualizing narrow arrays. Defaults to 1.
        title (str, optional): Title to give the plot. Defaults to None.
        vmin (float, optional): Minimum value for color mapping. Defaults to None.
        vmax (float, optional): Maximum value for color mapping. Defaults to None.
        norm (str, optional): Type of normalisation scale to use for this image. Defaults to None.
        verbose (int, optional): From 0 to 2. Specifies how many print statements to issue. Defaults to 2.

    Returns:
        figure, axis, image: matplotlib figure environment, axis object, and image mappable.
    """
    fig, ax = plt.subplots(figsize=(20, 25))
    if (norm == None or norm not in get_scale_names()):
        # Either they didn't specify, or they specified incorrectly. In either case, default to linear.
        if verbose == 2:
            print("Plot normalization unspecified or unrecognized, defaulting to 'linear'...")
        norm = 'linear'
    im = ax.imshow(array, aspect=aspect, norm=norm, origin="lower", vmin=vmin, vmax=vmax)
    if isinstance(aspect,str):
        plt.colorbar(mappable=im,aspect=40)
    else:
        plt.colorbar(mappable=im,fraction=min(0.5/aspect,0.15))
    ax.set_title(title)
    return fig, ax, im

def plot_fit(ax, t, lc, lc_err, t_interp, lc_interp, fit_color='red'):
    """Plots the fitted model over the data.

    Args:
        ax (_type_): _description_
        t (_type_): _description_
        lc (_type_): _description_
        lc_err (_type_): _description_
        t_interp (_type_): _description_
        lc_interp (_type_): _description_
    """
    if lc_err.size != 0:
        ax.errorbar(t, lc, yerr=lc_err, fmt='ko', capsize=3, zorder=0)
    else:
        ax.scatter(t, lc, color='k', zorder=0)
    
    ax.plot(t_interp, lc_interp, color=fit_color, zorder=2)
    return ax

def plot_res(ax, t, res, lc_err):
    """Plots the residuals on the given axis.

    Args:
        ax (_type_): _description_
        t (_type_): _description_
        res (_type_): _description_
        lc_err (_type_): _description_
    """
    if lc_err.size != 0:
        ax.errorbar(t, res, yerr=lc_err, fmt='ko', capsize=3, zorder=0)
    else:
        ax.scatter(t, res, color='k', zorder=0)

    ax.axhline(y=0,color='red',ls='--', zorder=2)
    return ax

def plot_corner(flat_samples, labels):
    fig = corner.corner(flat_samples, labels=labels)
    return fig

def plot_chains(ndim, samples, labels):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        try:
            ax = axes[i]
        except TypeError:
            # There is only one axis because there was only one sample.
            ax = axes
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    ax.set_xlabel("step number") # will automatically grab last used axis
    return fig, axes

def plot_post(ndim, samples, labels, n):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=False)
    for i in range(ndim):
        try:
            ax = axes[i]
        except TypeError:
            # There is only one axis because there was only one sample.
            ax = axes
        post = np.reshape(samples[:, :, i], (n))
        ax.hist(post, 100, alpha=0.3)
        ax.set_xlim(min(post), max(post))
        ax.set_ylabel(labels[i])
    return fig, axes

def plot_nest_post(ndim, samples, labels):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=False)
    for i in range(ndim):
        try:
            ax = axes[i]
        except TypeError:
            # There is only one axis because there was only one sample.
            ax = axes
        post = samples[:,i]
        ax.hist(post, 100, alpha=0.3)
        ax.set_xlim(min(post), max(post))
        ax.set_ylabel(labels[i])
    return fig, axes

def plot_allan(residuals):
    # Define maximum bin size.
    maxbins = len(residuals)//2

    # Retrieve rms from mc3.stats function.
    rms, rmslo, rmshi, res_std, bins = time_avg(residuals,
                                                maxbins=maxbins,
                                                binstep=1)
    
    # And plot everything.
    f = 1e-6 # norm factor for ppm
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(bins,y1=(rms-rmslo)/f,y2=(rms+rmshi)/f,color='k',
                    alpha=0.5,label='RMS Err',lw=1.5)
    ax.plot(bins, [i/f for i in res_std], c='red', lw=2.0, label="Gaussian Std Err")
    ax.plot(bins, [i/f for i in rms], c='k',label='RMS',lw=1.5)
    # Set up plot parmeters nicely and then return.
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("bin size [# ints]")
    ax.set_ylabel("rms [ppm]")
    return fig, ax