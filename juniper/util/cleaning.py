import numpy as np
from scipy import signal
from astropy.stats import sigma_clip

def median_spatial_filter(data, sigma, kernel):
    """Cleans one 2D array with median spatial filtering. 
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        data (np.array): 2D array of data.
        sigma (float): sigma threshold at which to reject outliers.
        kernel (tuple): tuple of two ints which must be odd. Kernal used \
            for median filtering.

    Returns:
        np.array: cleaned 2D array.
    """
    medfilt = signal.medfilt2d(data, kernel)
    diff = data - medfilt
    temp = sigma_clip(diff, sigma=sigma, axis=0)
    mask = temp.mask
    int_mask = mask.astype(float) * medfilt
    test = (~mask).astype(float)
    return (data*test) + int_mask

def median_timeseries_filter(data, sigma, kernel):
    """Cleans one 1D array with median spatial filtering. 
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        data (np.array): 1D array of data.
        sigma (float): sigma threshold at which to reject outliers.
        kernel (tuple): tuple of one ints which must be odd. Kernal used for median filtering.

    Returns:
        np.array: cleaned 1D array.
    """
    medfilt = signal.medfilt(data, kernel)
    diff = data - medfilt
    temp = sigma_clip(diff, sigma=sigma, axis=0)
    mask = temp.mask
    int_mask = mask.astype(float) * medfilt
    test = (~mask).astype(float)
    return (data*test) + int_mask

def colbycol_bckg(data, bckg, bckg_rows=[], trace_mask=None):
    """Performs column-by-column background subtraction on a given array.

    Args:
        data (np.array): 2D array of data, unchanged data.
        bckg (np.array): 2D array of data, data cleaned of outliers.
        bckg_rows (list, optional): list of integers which defines the background rows. Defaults to [].
        trace_mask (np.ma.masked_array, optional): mask to hide trace pixels with. Defaults to None.

    Returns:
        np.array: data array with column-by-column noise removed, and background of that noise.
    """
    # Define the background region using background rows and/or masks, if applicable.
    trace_mask[np.isnan(data)] = 1
    background_region = np.ma.masked_array(bckg,mask=trace_mask)
    if bckg_rows:
         background_region = background_region[bckg_rows, :]

    # Define the median background in each column and extend to a full-size array.
    background = np.ma.median(background_region, axis=0)
    background = np.array([background,]*data.shape[0])

    # And remove background from data.
    data -= background
    return data, background

def get_trace_mask(data, threshold=10000):
    """Build a mask using the given 2D data frame.

    Args:
        data (np.array): 2D array of data.
        threshold (float, optional): counts level at which to declare something
        as definitely part of the trace. Defaults to 10000.
    
    Returns:
        np.ma.mask: np mask hiding trace pixels.
    """
    # Determine statistics of region assumed to be background by count level.
    mu = np.median(data[data<threshold])
    sig = np.std(data[data<threshold])

    # If the data /positively/ exceeds the mean background level even by just
    # a small amount, it is definitely trace and must be masked.
    masked_fg = np.ma.masked_where(data - mu > 0.1*sig, data)
    return np.ma.getmask(masked_fg).astype(int)

def get_com_mask(data, width=5, upper=None, contrast=False):
    """Build a com mask using the given 2D data frame.

    Args:
        data (np.array): 2D array of data.
        width (int, optional): how many pixels from the COM to declare
        a pixel outside of the mask. Defaults to 5.
        upper (int, optional): for asymmetric masking - upper mask width.
        Defaults to None.
        contrast (bool, optional): a flag warning that this dataset suffers
        from low contrast between trace and background. Uses powers to
        amplify trace signal. Defaults to False.
    
    Returns:
        np.ma.mask: np mask hiding trace pixels.
    """
    # Amplify trace signal if needed.
    if contrast:
        data = data**9

    # Track the COM pixel in each column.
    pix_centers = np.arange(data.shape[0]) + 0.5
    COMs = signal.medfilt((np.sum(pix_centers[:,np.newaxis]*np.abs(data),axis=0)/np.sum(np.abs(data),axis=0)),7)
    COMs[np.isnan(COMs)] = 0.5*data.shape[0] # protect against nan columns, especially useful for G395H detector edge.

    # Polyfit to smooth over hiccups.
    x = np.array([i for i in range(len(COMs))])
    a,b,c = np.polyfit(x,COMs,deg=2)
    COMs = a*(x**2) + b*x + c

    # Integerize to use as indices in an array.
    integer_COMs = np.around(COMs - 0.5).astype(int)

    # Build a zero array and populate it with 1s where appropriate.
    trace_mask = np.zeros_like(data)

    for i, center in enumerate(integer_COMs):
        lb = center-width
        ub = center+width
        if upper:
            ub = center+upper
        if lb < 0:
            lb = 0
        if ub > trace_mask.shape[0] - 1:
            ub = trace_mask.shape[0] - 1
        trace_mask[lb:ub,i] = 1

    return trace_mask