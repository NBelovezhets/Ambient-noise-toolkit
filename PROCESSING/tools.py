import copy
import numpy as np 
import scipy
import itertools

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

def pws(cc_array, power=2):
    """
    Performs phase-weighted stack on array of time series.
    Follows methods of Schimmel and Paulssen, 1997. 
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack
    
    Originally written by Tim Clements
    Modified by Chengxin Jiang @Harvard (from NoisePy)
    
    Parameters
    ----------
    cc_array : numpy.ndarray
        N length array of time series data.
    power : int, optional
        Exponent for phase stack. The default is 2.

    Returns
    -------
    numpy.ndarray
        Phase weighted stack of time series data.

    """
    from scipy.signal import hilbert
    from scipy.fftpack import next_fast_len
    
    N,M = cc_array.shape

    # construct analytical signal
    analytic = hilbert(cc_array, axis=1, N=next_fast_len(M))[:,:M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j*phase),axis=0)
    phase_stack = np.abs(phase_stack)**(power)

    # weighted is the final waveforms
    weighted = np.multiply(cc_array,phase_stack)
    return np.mean(weighted,axis=0)

def rotation(Greens_tensor, az, backaz):
    """
    Rotate the Green's tensor from a E-N-Z system into a R-T-Z one

    Parameters
    ----------
    Greens_tensor : numpy.ndarray size of (9, N)
        9 component Green's tensor in E-N-Z system
        ['EE','EN','EZ','NE','NN','NZ','ZE','ZN','ZZ']
    az : float
        Azimuth in degrees
    backaz : float
        Back azimuth in degrees

    Returns
    -------
    tcorr : numpy.ndarray size of (9, N), dtype=np.float32
        9 component Green's tensor in R-T-Z system
        ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']

    """
    pi = np.pi
    azi = az
    baz = backaz
    
    ncomp, npts = Greens_tensor.shape

    cosa = np.cos(azi*pi/180)
    sina = np.sin(azi*pi/180)
    cosb = np.cos(baz*pi/180)
    sinb = np.sin(baz*pi/180)

    # rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    tcorr = np.zeros(shape=(9,npts),dtype=np.float32)
    tcorr[0] = -cosb*Greens_tensor[7]-sinb*Greens_tensor[6]
    tcorr[1] = sinb*Greens_tensor[7]-cosb*Greens_tensor[6]
    tcorr[2] = Greens_tensor[8]
    tcorr[3] = -cosa*cosb*Greens_tensor[4]-cosa*sinb*Greens_tensor[3]-sina*cosb*Greens_tensor[1]-sina*sinb*Greens_tensor[0]
    tcorr[4] = cosa*sinb*Greens_tensor[4]-cosa*cosb*Greens_tensor[3]+sina*sinb*Greens_tensor[1]-sina*cosb*Greens_tensor[0]
    tcorr[5] = cosa*Greens_tensor[5]+sina*Greens_tensor[2]
    tcorr[6] = sina*cosb*Greens_tensor[4]+sina*sinb*Greens_tensor[3]-cosa*cosb*Greens_tensor[1]-cosa*sinb*Greens_tensor[0]
    tcorr[7] = -sina*sinb*Greens_tensor[4]+sina*cosb*Greens_tensor[3]+cosa*sinb*Greens_tensor[1]-cosa*cosb*Greens_tensor[0]
    tcorr[8] = -sina*Greens_tensor[5]+cosa*Greens_tensor[2]

    return tcorr

def demean(data):
    """
    Demean signal

    Parameters
    ----------
    data : numpy.ndarray
        Signal to detrend.

    Returns
    -------
    data : numpy.ndarray
        Demeaned signal.

    """
    data = copy.deepcopy(data)
    mean = np.mean(data)
    data -= mean
    return data

def detrend(data):
    '''
    Delete trends (linear and polynomial - 2 and 3 degree) from the waveform

    Parameters
    ----------
    data : numpy.ndarray
        Signal to detrend.

    Returns
    -------
    data : numpy.ndarray
        Detrended signal.

    '''
    data = scipy.signal.detrend(data, axis=- 1, type='linear', bp=0, overwrite_data=False)
    x = np.arange(len(data))
    fit = np.polyval(np.polyfit(x, data, deg=3), x)
    data -= fit
    x = np.arange(len(data))
    fit = np.polyval(np.polyfit(x, data, deg=2), x)
    data -= fit
    return data

def bandpass_filter(data, freqmin, freqmax, corners, dt):
    '''
    Bandpass filtering of the signal

    Parameters
    ----------
    data : numpy.ndarray
        Signal to filter.
    freqmin : float
        Minimum frequency of the bandpass filter in Hz.
    freqmax : TYPE
        Maximum frequency of the bandpass filter in Hz.
    corners : int
        The order of the filter.
    dt : float
        Discretization step.

    Returns
    -------
    filt_signal : numpy.ndarray
        Filtered signal.
    '''
    import scipy.signal as signal
    df = 1/dt
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    z, p, k = signal.iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    
    sos = signal.zpk2sos(z, p, k)
    
    firstpass = signal.sosfilt(sos, data)
    filt_signal = signal.sosfilt(sos, firstpass[::-1])[::-1]
    return filt_signal

def one_bit(data):
    """
    Apply one-bit time domain normalization - save only a sign of the signal.
    See Bensen et al., 2007

    Parameters
    ----------
    data : numpy.ndarray
        Signal to normalize.

    Returns
    -------
    signal : numpy.ndarray
        Normalized signal.

    """
    trace = copy.deepcopy(data)
    trace = np.sign(trace)
    
    return trace

#vectorized function one_bit
vectorize_one_bit = np.vectorize(one_bit, signature="(n)->(n)")

def clip(data, lim):
    '''
    Apply Clipping - time domain normalization. Values of the signal that are more/less than a threshold (maximum/minimum of the signal = lim * std(signal)) become equated to the threshold.
    See Bensen et al., 2007

    Parameters
    ----------
    data : numpy.ndarray
        Signal to normalize.
    lim : int
        Value to compute the threshold - multiply by a standard deviation of the signal (the square root of the average of the squared deviations from the mean).

    Returns
    -------
    trace : numpy.ndarray
        Normalized signal.

    '''
    trace = copy.deepcopy(data)
    std = np.std(trace)
    lim_v = lim*std
    
    for i in trace:
        if i>lim_v or i<-lim_v:
            i = lim_v
            
    return trace

#vectorized function clip
vectorize_clip = np.vectorize(clip, signature="(n),()->(n)")

def reject(data, lim = 3):
    '''
    Apply Rejecting - time domain normalization. Values of the signal that are more/less than a threshold (maximum/minimum of the signal = lim * std(signal)) become zeroes.

    Parameters
    ----------
    data : numpy.ndarray
        Signal to normalize.
    lim : int, optional
        Value to compute the threshold - multiply by a standard deviation of the signal (the square root of the average of the squared deviations from the mean).
        The default is 3.

    Returns
    -------
    trace : numpy.ndarray
        Normalized signal.

    '''
    trace = copy.deepcopy(data)
    std = np.std(trace)
    lim = lim*std
    trace[trace > lim] = 0
    trace[trace < -lim] = 0
    return trace

#vectorized reject
vectorize_reject = np.vectorize(reject, signature="(n),()->(n)")

def reject_after(data, lim, number, dt):
    '''
    Apply Rejecting - time domain normalization. Values of the signal that are more/less than a threshold (maximum/minimum of the signal = lim * std(signal)) become zeros. Also number of seconds after values become zeros.

    Parameters
    ----------
    data : numpy.ndarray
        Signal to normalize.
    lim : int
        Value to compute the threshold - multiply by a standard deviation of the signal (the square root of the average of the squared deviations from the mean).
    number : int
        How many seconds you want to become zeros after the signal value to reject.
    dt : float
        Discretization step
    
    Returns
    -------
    trace : numpy.ndarray
        Normalized signal.

    '''
    trace = copy.deepcopy(data)
    std = lim*np.std(trace)
    for i in range(len(trace)):
        if trace[i] > std or trace[i] < -std:
            for j in range(np.ceil(number/(dt)).astype('int')):
                trace[i+j]=0
    return trace

#vectorized reject_after
vectorize_reject = np.vectorize(reject_after, signature="(n),(),()->(n)")

def reject_around(data, lim, number_sec, dt):
    '''
    Apply Rejecting - time domain normalization. Values of the signal that are more/less than a threshold (maximum/minimum of the signal = lim * std(signal)) become zeros. Also number of seconds around values become zeros.

    Parameters
    ----------
    data : numpy.ndarray
        Signal to normalize.
    lim : int
        Value to compute the threshold - multiply by a standard deviation of the signal (the square root of the average of the squared deviations from the mean).
    number_sec : int
        How many seconds you want to become zeros around the signal value to reject.
    dt : float
        Discretization step

    Returns
    -------
    trace : numpy.ndarray
        Normalized signal.

    '''
    trace = copy.deepcopy(data)
    std = np.std(trace)
    lim_v = lim*std
    number = np.ceil(number_sec/(dt)).astype('int')
    
    for i in range(0, number):
    
        if trace[i]>lim_v or trace[i]<-lim_v:
            trace[0:i+number] = 0
    
    for i in range(number, len(trace)-number):
        
        if trace[i]>lim_v or trace[i]<-lim_v:
            trace[i-number:i+number] = 0
            
    for i in range(len(trace)-number, len(trace)):
    
        if trace[i]>lim_v or trace[i]<-lim_v:
            trace[i:len(trace)] = 0
            
    return trace

#vectorized reject_around
vectorize_reject = np.vectorize(reject_around, signature="(n),(),()->(n)")

def ram_adoptive(sig, freq_min, freq_max, dt, corners, lwin):
    '''
    Apply Adoptive RAM(Running absolute mean) - time domain normalization. 
    See Bensen et al., 2007
    Running absolute mean normalization whereby the filtered (bandpass) waveform is normalized by a running average of its absolute value.   

    Parameters
    ----------
    sig : numpy.ndarray
        Signal to normalize.
    freq_min : float
        Minimum frequency of the bandpass filter in Hz.
    freq_max : float
        Maximum frequency of the bandpass filter in Hz.
    dt : float
        Discretization step.
    corners : int
        The order of the filter.
    lwin : float
        Lenght of the window in seconds to compute running average of its absolute value.

    Returns
    -------
    numpy.ndarray
        Normalized signal.

    '''
    import scipy.signal as signal
    df = 1/dt
    fe = 0.5 * df
    low = freq_min / fe
    high = freq_max / fe
    len_signal = len(sig)
    sig*= cosine_taper(len_signal, p = 0.01, sactaper=True, halfcosine=False)
    
    z, p, k = signal.iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    
    sos = signal.zpk2sos(z, p, k)
    
    firstpass = signal.sosfilt(sos, sig)
    filt_signal = signal.sosfilt(sos, firstpass[::-1])[::-1]

    lwin = int(lwin/dt)
    
    s = np.concatenate((filt_signal[-lwin//2:], filt_signal, filt_signal[:lwin//2]))

    s_mat = np.vstack([s[i:i+lwin] for i in range(0, len_signal, 1)])
    
    w_array = np.zeros_like(sig)
    
    w_array = np.sum(np.abs(s_mat), axis=-1)/(2. * lwin +1)
    w_array[np.where(w_array == 0)] = 1  
    
    return sig/w_array

#vectorized ram_adoptive
vectorize_ram_adop = np.vectorize(ram_adoptive, signature="(n),(),(),(),(),()->(n)")

def ram(sig, dt, lwin):
    '''
    Apply RAM(Running absolute mean) - time domain normalization. 
    See Bensen et al., 2007
    Running absolute mean normalization whereby the waveform is normalized by a running average of its absolute value.   

    Parameters
    ----------
    sig : numpy.ndarray
        Signal to normalize.
    dt : float
        Discretization step.
    lwin : float
        Lenght of the window in seconds to compute running average of its absolute value.

    Returns
    -------
    numpy.ndarray
        Normalized signal.

    '''
    len_signal = len(sig)
    
    lwin = int(lwin/dt)
    
    s = np.concatenate((sig[-lwin//2:], sig, sig[:lwin//2]))

    s_mat = np.vstack([s[i:i+lwin] for i in range(0, len_signal, 1)])
    
    w_array = np.zeros_like(sig)
    
    w_array = np.sum(np.abs(s_mat), axis=-1)/(2. * lwin +1)
    w_array[np.where(w_array == 0)] = 1  
    
    return sig/w_array

#vectorized ram
vectorize_ram = np.vectorize(ram, signature="(n),(),()->(n)")

def creat_invert_resp(trace, inventory):
    '''
    Create inverted instrumental response

    Parameters
    ----------
    trace : Class '~obspy.core.trace.Trace'
        An object containing data of a continuous series, such as a seismic trace.
    inventory : class:`~obspy.core.inventory.inventory.Inventory`
            or :class:`~obspy.core.inventory.network.Network` or a list
            containing objects of these types or a string with a filename of
            a StationXML file.
        Station metadata to use in search for response for
            each trace in the stream.

    Returns
    -------
    freq_response : numpy.ndarray
        Inverted frequency response under water-level of max spectrum amplitude
    freqs : numpy.ndarray
        Corresponding frequencies

    '''
    from obspy.signal.invsim import invert_spectrum
    response = trace._get_response(inventory)
        
    from obspy.signal.util import _npts2nfft
    nfft = _npts2nfft(len(trace.data))
    freq_response, freqs = \
            response.get_evalresp_response(trace.stats.delta, nfft,
                                           output="Vel")
    invert_spectrum(freq_response, 60)
    
    return freq_response, freqs

def remove_response(trace, freq_response, freqs, prefilt, taper_fraction=0.05):
    '''
    Remove instumental response

    Parameters
    ----------
    trace : numpy.ndarray
        Trace to remove response.
    freq_response : numpy.ndarray
        Inverted frequency response
    freqs : numpy.ndarray
        Corresponding frequencies
    prefilt : list or tuple(float, float, float, float)
        Apply a bandpass filter in frequency domain to the
        data before deconvolution. The list or tuple defines
        the four corner frequencies `(f1, f2, f3, f4)` of a cosine taper
        which is one between `f2` and `f3` and tapers to zero for
        `f1 < f < f2` and `f3 < f < f4`.
    taper_fraction : float, optional
        Taper fraction of cosine taper to use.. The default is 0.05.

    Returns
    -------
    data : numpy.ndarray
        Trace which instrumental response removed.

    '''
    from obspy.signal.invsim import cosine_taper, cosine_sac_taper
    from obspy.signal.util import _npts2nfft
            
    data = copy.deepcopy(trace)
    freq_response = freq_response
    freqs = freqs
    pre_filt = prefilt
    
    npts = len(data)
    
    mean = np.ceil(np.mean(data)).astype("int32")
    data -= mean
    data = data.astype('float64')
    data *= cosine_taper(npts, taper_fraction, sactaper=True, halfcosine=False)
    
    nfft = _npts2nfft(len(data))
    data = np.fft.rfft(data, n=nfft)
    
    freq_domain_taper = cosine_sac_taper(freqs, flimit=pre_filt)
    data *= freq_domain_taper
    
    data *= freq_response
    data[-1] = abs(data[-1]) + 0.0j
    data = np.fft.irfft(data)[0:npts]
    
    return data
    
def compute_abs_spectrum(data, dt):
    '''
    Compute the absolute one-dimensional discrete Fourier Transform for real input.

    This function computes the absolute one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    data : numpy.ndarray
        Signal to compute spectrum.
    dt : int
        Discretization step

    Returns
    -------
    fft_spectr : numpy.ndarray
        The truncated or zero-padded input, transformed along the axis indicated by axis, 
        or the last one if axis is not specified. If n is even, the length of the transformed axis is (n/2)+1. 
        If n is odd, the length is (n+1)/2.
    freqs : numpy.ndarray
        Array of length n//2 + 1 containing the sample frequencies.

    '''
    fft_spectr = abs(np.fft.rfft(data*cosine_taper(len(data))))
    freqs = np.fft.rfftfreq(len(data), dt)
    return fft_spectr, freqs

def whitening(data, dt, freqmin, freqmax, whiten_type):
    '''
    Apply whitening - frequency domain normalization.
    Two types:
        PSD - Inversely weighting the complex spectrum by a smoothed version of the amplitude spectrum
        using a square root of the power spectral density.
        HANN - Inversely weighting the complex spectrum by the amplitude spectrum in a Hann window.
        Cosine - Inversely weighting the complex spectrum by the amplitude spectrum in a cosine window.
        
    Parameters
    ----------
    data : numpy.ndarray
        Signal to normalize.
    dt : float
        Discretization step.
    freqmin : float
        Minimum frequency: frequencies that are less than it become zeros.
    freqmax : float
        Maximum frequency: frequencies that are more than it become zeros.
    whiten_type : string
        PSD, HANN or Cosine.

    Returns
    -------
    numpy.ndarray
        Normalized signal

    '''
    
    from scipy.stats import scoreatpercentile
    nfft = data.shape[0]

    Nfft = int(nfft)

    data = copy.deepcopy(data)

    Napod = 100

    fft_ = np.fft.fft(data, Nfft, axis=-1)

    freqVec = np.fft.fftfreq(Nfft, d=dt)[:Nfft // 2]

    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]

    low = J[0] - Napod
    if low <= 0:
        low = 0

    porte1 = J[0]
    porte2 = J[-1]

    high = J[-1] + Napod

    if high > Nfft / 2:
        high = int(Nfft // 2)
    
    if whiten_type == "PSD":
        
        taper = np.ones(Nfft // 2 + 1)
        taper[0:low] *= 0
        taper[low:porte1] *= np.cos(np.linspace(np.pi / 2., 0, int(porte1 - low))) ** 2
        taper[porte2:high] *= np.cos(np.linspace(0., np.pi / 2., int(high - porte2))) ** 2
        taper[high:] *= 0
        taper *= taper
        
        freqs, pxx = scipy.signal.welch(data, fs=1/dt, nperseg=Nfft, detrend='constant')
    
        psds = np.sqrt(pxx)
        psds = np.where(psds == 0., 2e-24, psds)

        fft_[:Nfft // 2 + 1] /= psds
        fft_[:Nfft // 2 + 1] *= taper
    
        tmp = fft_[porte1:porte2]
        imin = scoreatpercentile(tmp, 5)
        imax = scoreatpercentile(tmp, 95)
        not_outliers = np.where((tmp >= imin) & (tmp <= imax))[0]
        rms = tmp[not_outliers].std() * 1.0
        
        A = np.abs(fft_[int(porte1):int(porte2)])
        fft_[int(porte1):int(porte2)]/=A
        np.clip(A, -rms, rms, A)  # inplace
        fft_[int(porte1):int(porte2)]*=A        
        fft_[0:low] *= 0
        fft_[high:] *= 0
        
    elif whiten_type == "HANN":
        
        hann = np.hanning(int(porte2 - porte1 + 1))  # / float(porte2-porte1)
        fft_ /= (np.abs(fft_) + 2e-24)
        fft_[:porte1] *= 0.0
        fft_[porte1:porte2 + 1] *= hann
        fft_[porte2 + 1:] *= 0.0
        
    elif whiten_type == 'Cosine':
        cos = cosine_taper(int(porte2 - porte1 + 1), p = 0.01, sactaper=True, halfcosine=False)
        fft_ /= (np.abs(fft_) + 2e-24)
        fft_[:porte1] *= 0.0
        fft_[porte1:porte2 + 1] *= cos
        fft_[porte1:porte2+1] = np.exp(1j * np.angle(fft_[porte1:porte2+1]))        
        fft_[porte2 + 1:] *= 0.0
        
    else:
        return('No such type of whitening')
    
    # Hermitian symmetry (because the input is real)
    fft_[-(Nfft // 2) + 1:] = np.conjugate(fft_[1:(Nfft // 2)])[::-1]

    return np.real(np.fft.ifft(fft_, Nfft, axis=-1))

def whitening_NE(signalN, signalE, dt, freqmin, freqmax):
    '''
    Apply whitening for N and E components of the seismic trace - frequency domain normalization.
    
    HANN - Inversely weighting the complex spectrum by the average of the summ of amplitude spectrums.

    Parameters
    ----------
    signalN : numpy.ndarray
        Array of trace N.
    signalE : numpy.ndarray
        Array of trace E.
    dt : float
        Discretization step.
    freqmin : float
        Minimum frequency: frequencies that are less than it become zeros.
    freqmax : float
        Maximum frequency: frequencies that are more than it become zeros.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Normalized N and E components.

    '''
    dataN = signalN
    dataE = signalE
    dt = dt
    freqmin = freqmin
    freqmax = freqmax
    
    nfft = dataN.shape[0]
    Nfft = int(nfft)

    Napod = 100

    fft_N = np.fft.fft(dataN, Nfft, axis=-1)
    fft_E = np.fft.fft(dataE, Nfft, axis=-1)

    freqVec = np.fft.fftfreq(Nfft, d=dt)[:Nfft // 2]

    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]

    low = J[0] - Napod
    if low <= 0:
        low = 0

    porte1 = J[0]
    porte2 = J[-1]

    high = J[-1] + Napod

    if high > Nfft / 2:
        high = int(Nfft // 2)
           
    hann = np.hanning(int(porte2 - porte1 + 1))  # / float(porte2-porte1)
    
    fft_N /= (np.abs(fft_N) + np.abs(fft_E) + 2e-24)/2
    fft_E /= (np.abs(fft_E) + np.abs(fft_N) + 2e-24)/2
    fft_N[:porte1] *= 0.0
    fft_N[porte1:porte2 + 1] *= hann
    fft_N[porte2 + 1:] *= 0.0
    fft_E[:porte1] *= 0.0
    fft_E[porte1:porte2 + 1] *= hann
    fft_E[porte2 + 1:] *= 0.0

    # Hermitian symmetry (because the input is real)
    fft_N[-(Nfft // 2) + 1:] = np.conjugate(fft_N[1:(Nfft // 2)])[::-1]
    fft_E[-(Nfft // 2) + 1:] = np.conjugate(fft_E[1:(Nfft // 2)])[::-1]
    
    return np.real(np.fft.ifft(fft_N, Nfft, axis=-1)), np.real(np.fft.ifft(fft_E, Nfft, axis=-1))

def compute_cross_corr(pair_trace, blocks, steps, fs, len_save):
    '''
    Compute cross-correlation between two signals in the moving window with the step.
    Save part of the cross-correlation that are symmetrical with respect to zero.

    Parameters
    ----------
    pair_trace : numpy.ndarray(2, N) or list(numpy.ndarray, numpy.ndarray)
        2 signals to cross-correlate.
    blocks : int
        Moving window width in seconds.
    steps : int
        Step width in seconds.
    fs : float
        Sampling frequency.
    len_save : int
        Lenght of the saving cross-correlation in seconds.

    Returns
    -------
    res : numpy.ndarray
        Cross-correlation between two signals from -len_save to +len_save.

    '''
    width = int(fs*blocks)
    width_save = int(fs*len_save)
    stepsize = int(fs*steps)
    n = len(pair_trace[0])
    
    cos_tap_window = cosine_taper(width)
    
    trace_1 = np.vstack([pair_trace[0][i:i+width] for i in range(0,n-stepsize, stepsize)])
    trace_2 = np.vstack([pair_trace[1][i:i+width] for i in range(0,n-stepsize, stepsize)])
    trace_2 = np.flip(trace_2, axis = 1)      
    trace_1 *= cos_tap_window
    trace_2 *= cos_tap_window

    res = scipy.signal.fftconvolve(trace_1, trace_2, mode="same", axes=1)
    
    energy = np.linalg.norm(res, axis = 1)
    res = (res.T/energy).T
    res = np.sum(res, axis = 0)

    res = res[len(res)//2 - width_save:len(res)//2 + width_save + 1]
    return res

def compute_cross_corr_with_whiten(pair_trace, blocks, steps, fs, len_save, freqmin, freqmax):
    '''
    Compute cross-correlation between two whitened signals in the moving window with the step.
    Save part of the cross-correlation that are symmetrical with respect to zero.
    Before cross-correlating apply to signals detrending and whitening in the window from freqmin to freqmax (inversely weighting the complex spectrum by the amplitude spectrum).
    
    Parameters
    ----------
    pair_trace : numpy.ndarray(2, N) or list(numpy.ndarray, numpy.ndarray)
        2 signals to cross-correlate.
    blocks : int
        Moving window width in seconds.
    steps : int
        Step width in seconds.
    fs : float
        Sampling frequency.
    len_save : int
        Lenght of the saving cross-correlation in seconds.
    freqmin : float
        Minimum frequency: frequencies that are less than it become zeros.
    freqmax : float
        Maximum frequency: frequencies that are more than it become zeros.

    Returns
    -------
    res : numpy.ndarray
        Cross-correlation between two signals from -len_save to +len_save.

    '''
    width = int(fs*blocks)
    width_save = int(fs*len_save)
    stepsize = int(fs*steps)
    n = len(pair_trace[0])
    cos_tap_window = cosine_taper(fs*blocks)
    
    trace_1 = np.vstack([pair_trace[0][i:i+width] for i in range(0,n-stepsize, stepsize)])
    trace_2 = np.vstack([pair_trace[1][i:i+width] for i in range(0,n-stepsize, stepsize)])     
    trace_1 *= cos_tap_window
    trace_2 *= cos_tap_window
    
    trace_1 = list(map(detrend, trace_1))
    trace_2 = list(map(detrend, trace_2))
    
    trace_1 = whitening(trace_1, 1/fs, freqmin, freqmax, whiten_type="HANN")
    trace_2 = whitening(trace_2, 1/fs, freqmin, freqmax, whiten_type="HANN")
    trace_2 = np.flip(trace_2, axis = 1) 
    
    res = scipy.signal.fftconvolve(trace_1, trace_2, mode="same", axes=1)
    
    energy = np.linalg.norm(res, axis = 1)
    res = (res.T/energy).T
    res = np.sum(res, axis = 0)

    res = res[len(res)//2 - width_save:len(res)//2 + width_save + 1]
    
    return res

def make_combinations_of_traces(data):
    '''
    Create an array that consists of all variable combinations between all signals.  
    For example: if you have an array [trace1, trace2, trace3], the function returns 
    an array [[trace1, trace2], [trace1, trace3], [trace2, trace3]]
    Parameters
    ----------
    data : numpy.ndarray(M, N) or list(M arrays)
        Array of M signals with lenght N.

    Returns
    -------
    trace_list : numpy.ndarray(M*(M-1)/2 lists (array(N), array(N)))
        Array of all variable combinations between signals.

    '''
    combinations = list(itertools.combinations(list(range(len(data))), 2))
    trace_list = np.array([[data[i[0]], data[i[1]]] for i in combinations])
    
    return trace_list
