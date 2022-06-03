import matplotlib.colors as cmat
import matplotlib.pyplot as plt
import numpy as np

def plot_trace(trace, t):
    '''
    Plot signals in time

    Parameters
    ----------
    trace : numpy.ndarray
        Signal to draw.
    t : numpy.ndarray
        Time of the signal.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(30, 5))
    plt.plot(t, trace, color = 'black')
    plt.tick_params(axis='x', which='major', labelsize=22)
    plt.tick_params(axis='y', which='major', labelsize=22)
    plt.xlabel('Time, sec', fontsize=24)
    plt.show()

def plot_cross_correlate(summ, t, names_stat, xlim=(-200, 200)):
    '''
    Plot cross-correlation in xlim limits.

    Parameters
    ----------
    summ : numpy.ndarray
        Cross-correlation.
    t : numpy.ndarray
        Time of the cross-correlation.
    names_stat : string
        Names of the pair of stations between which the cross-correlation was computed.
    xlim : tuple(int, int), optional
        Limits in time of plotting the cross-correlation. The default is (-200, 200).

    Returns
    -------
    None.

    '''        
    plt.ioff()
    plt.figure(figsize=(30, 5))
    plt.plot(t, summ, color = 'black')
    plt.xlim(xlim[0], xlim[1])
    plt.title('Cross-correlation between stations ' + names_stat , fontsize=28)
    plt.tick_params(axis='x', which='major', labelsize=22)
    plt.tick_params(axis='y', which='major', labelsize=22)
    plt.xlabel('Lag time, sec', fontsize=24)
    plt.show()

def plot_all_cross_correlate(time, t, data, names_stat, t_lim = (-200, 200), norm = False):
    '''
    Plot all daily cross-correlations.

    Parameters
    ----------
    time : array of dates, for example, UTC datetime 
        Dates for daily cross-correlations.
    t : numpy.ndarray
        Time of the cross-correlations.
    data : array of the numpy.ndarray
        Array of all daily cross-correlations.
    names_stat : string
        Names of the pair of stations between which the cross-correlation was computed.
    t_lim : tuple(int, int), optional
        Limits in time of plotting the cross-correlation. The default is (-200, 200).
    norm : bool, optional
        If true, cross-correlations will norm between -1, 1. The default is False.

    Returns
    -------
    None.

    '''
    X, Y = np.meshgrid(time, np.flip(t, 0))
    data = np.vstack(data).T
    
    plt.ioff()
    fig, ax = plt.subplots(1,1, figsize=(20,12))
    
    if norm:
        cm = ax.pcolormesh(X, Y, data, norm = cmat.TwoSlopeNorm(vcenter = 0, vmin = -1, vmax = 1), cmap="seismic")
    
    else:
        cm = ax.pcolormesh(X, Y, data, norm = cmat.TwoSlopeNorm(vcenter = 0), cmap="seismic")
    
    cb = plt.colorbar(cm, ax=ax)
    
    for l in cb.ax.get_yticklabels():
        l.set_fontsize(20)
    
    ax.tick_params(axis='x', which='major', labelsize=20, labelrotation=-45)
    ax.tick_params(axis='y', which='major', labelsize=20)
    ax.set_ylim(t_lim[0], t_lim[1])
    ax.grid(visible = True, which="major", axis='x', color='grey', linestyle='-')
    ax.set_title('Cross-correlation between ' + names_stat + " " + ' from ' + str(time[0]) + ' to ' + str(time[-1]) , fontsize = 24)
    ax.set_ylabel('Lag time, sec', fontsize=22)
    ax.set_xlabel('Date', fontsize=22)
    
    plt.show()

def plot_spectrum(data, dt, yscale = True, xscale = False, xlim = (0, 5)):
    '''
    Plot absolute spectrum

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum of a signal.
    freqs : numpy.ndarray
        Frequencies of the Fourier spectrum.
    yscale : bool, optional
        If True, yscale = symlog. The default is True.
    xscale : bool, optional
        If True, xscale = log. The default is False.
    xlim : tuple(int, int), optional
        Limits of the frequencies for plotting. The default is (0, 5).

    Returns
    -------
    None.

    '''
    
    def cosine_taper(npts, p=0.1, halfcosine=True, sactaper=False):
        """
        Make a cosine taper.

        Parameters
        ----------
        npts : int
            Number of points of cosine taper.
        p : float, optional
            Decimal percentage of cosine taper (ranging from 0 to 1). Default
            is 0.1 (10%) which tapers 5% from the beginning and 5% form the end.
            The default is 0.1.
        halfcosine : bool, optional
            If True the taper is a half cosine function. If False it is a quarter cosine function.
            The default is True.
        sactaper : bool, optional
            If set to True the cosine taper already tapers at the corner frequency (SAC behavior). 
            By default, the taper has a value of 1.0 at the corner frequencies.
            The default is False.

        Returns
        -------
        cos_win : float NumPy :class:`~numpy.ndarray`
            Cosine taper array/vector of length npts.

        """
        
        if p == 0.0 or p == 1.0:
            frac = int(npts * p / 2.0)
        else:
            frac = int(npts * p / 2.0 + 0.5)

        idx1 = 0
        idx2 = frac - 1
        idx3 = npts - frac
        idx4 = npts - 1
        
        if sactaper:
            # in SAC the second and third
            # index are already tapered
            idx2 += 1
            idx3 -= 1

        # Very small data lengths or small decimal taper percentages can result in
        # idx1 == idx2 and idx3 == idx4. This breaks the following calculations.
        if idx1 == idx2:
            idx2 += 1
        if idx3 == idx4:
            idx3 -= 1

        # the taper at idx1 and idx4 equals zero and
        # at idx2 and idx3 equals one
        cos_win = np.zeros(npts)
        
        if halfcosine:
            
            cos_win[idx1:idx2 + 1] = 0.5 * (
                1.0 - np.cos((np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) /
                              (idx2 - idx1))))
            cos_win[idx2 + 1:idx3] = 1.0
            cos_win[idx3:idx4 + 1] = 0.5 * (
                1.0 + np.cos((np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) /
                              (idx4 - idx3))))
        else:
            cos_win[idx1:idx2 + 1] = np.cos(-(
                np.pi / 2.0 * (float(idx2) -
                               np.arange(idx1, idx2 + 1)) / (idx2 - idx1)))
            cos_win[idx2 + 1:idx3] = 1.0
            cos_win[idx3:idx4 + 1] = np.cos((
                np.pi / 2.0 * (float(idx3) -
                               np.arange(idx3, idx4 + 1)) / (idx4 - idx3)))

        # if indices are identical division by zero
        # causes NaN values in cos_win
        if idx1 == idx2:
            cos_win[idx1] = 0.0
        if idx3 == idx4:
            cos_win[idx3] = 0.0
            
        return cos_win
    
    nfft = len(data)
    spectrum = abs(np.fft.fft(data*cosine_taper(len(data), p = 0.001)))[:nfft//2]
    freqs = np.fft.fftfreq(len(data), dt)[:nfft//2]
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Verdana']
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.plot(freqs, spectrum, color="black")
    
    if yscale:
        ax.set_yscale('symlog')
    
    if xscale:
        ax.set_xscale('log')
        
    ax.set_xlabel("Frequency, Hz", fontsize=24)
    ax.set_ylabel("Amplitude", fontsize=24)
    ax.set_xlim(xlim[0], xlim[1])
    
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)
    plt.title('Absolute spectrum', fontsize=24)
    plt.show()