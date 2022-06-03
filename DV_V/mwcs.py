
import numpy as np
import scipy
from numba import jit
from progress.bar import IncrementalBar
import scipy.fftpack as sf
import scipy.signal
import multiprocessing 
scipy.fft.set_workers(multiprocessing.cpu_count())



@jit(nopython=True, fastmath=True)
def linear_detrend(data):
  
    new_data = data.copy()
   
    dshape = data.shape
    N = dshape[-1]

    
    A = np.ones((N, 2), np.float64)
    A[:, 0] = np.arange(1, N + 1) * 1.0 / N
    
    coef, resids, rank, s = np.linalg.lstsq(A, data)
    new_data -= np.dot(A, coef)
  
        
    # Put data back in original shape.
    ret = np.reshape(new_data, dshape )
  
    return ret








@jit(nopython=True)
def nextpow2(x):
    return np.ceil(np.log2(np.abs(x)))


                         
def extract_dtt(current, reference, freq_min, freq_max, df, tmin, window_length, window_step,
         hanningwindow):
    """
    Estimate relative time shifts as the first part of multiple window cross-spectrum
    analysis of two waveform according to Clarke et al., 2011.

    Parameters
    ----------
    current : float NumPy :class:`~numpy.ndarray`
        Current waveform (size N) to be compared  with reference waveform.
    reference : float NumPy :class:`~numpy.ndarray`
        Reference waveform (size N) to be compared with current waveform.
    freq_min : float
        Minimum frequency of used signal's passband.
    freq_max : float
        Maximum frequency of used signal's passband.
    df : float
        Sampling rate of used signals.
    tmin : float
        The leftmost record time of used signals.
    window_length : float
        Length of moving window to signals comparing.
        It is recommended to use windowlength at least 2 maximum periods to correct phase shift estimation at low periods.
    window_step : float
        Step of moving window to signals comparing.
        It is recommended to use window_step about 2 minimum periods to optimising perfomance time.
    hanningwindow : float NumPy :class:`~numpy.ndarray`
        Hanning smothing window (size M).
        
    Returns
    -------
    rec_time : float NumPy :class:`~numpy.ndarray`
        Record's time of centers of moving windows.
    time_shift : float NumPy :class:`~numpy.ndarray`
        Time shifts correspond to each window.
    time_shift_err : float NumPy :class:`~numpy.ndarray`
        Uncertainty of each time shift value.
    mean_coh : float NumPy :class:`~numpy.ndarray`
        Mean coherency in each window.

    """

    # Find lenght of moving window in samples
    window_length_samples = np.int32(window_length * df)
    
    # Find the closest power of 
    padd = int(2 ** (nextpow2(window_length_samples) + 2))



    # Create a cosine taper for signal's smothing prior Fourier
    taper = cosine_taper(window_length_samples, 0.85)

    # Find the ending indexes of each window
    maxinds = np.arange(window_length_samples, len(current), int(window_step*df))
    # Find the beginig's indexes of each window
    mininds = np.arange(0, len(current) - window_length_samples, int(window_step*df))

    # Create arrays to save values of time shift, its error, mean corency and central time in each window
    time_shift = np.zeros_like(maxinds,  dtype=np.float64)
    time_shift_err = np.zeros_like(maxinds, dtype=np.float64)
    mean_coh = np.zeros_like(maxinds,  dtype=np.float64)
    rec_time = np.zeros_like(maxinds,  dtype=np.float64)
    
    # Find the number of loops
    cycles = len(maxinds)

    # Perform processing in each moving window
    for i in range(cycles):
        
        # Get window's begin ann end indexes
        minind = mininds[i]
        maxind = maxinds[i]
        
        # Get windowed signals of current and reference wavefrom, detrend it and apply taper 
        cci = current[minind:maxind]
        cci = linear_detrend(cci)
        cci *= taper
        
        cri = reference[minind:maxind]
        cri = linear_detrend(cri)
        cri *= taper

        
        
        # Compute spectrum of current and reference windowed signal
        fcur = sf.fft(cci, n=padd)[:padd // 2]
        fref = sf.fft(cri, n=padd)[:padd // 2]
    
        # Compute the amplitude spectrum square
        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2
        
        fref2 = fref2.astype(np.float64)
        fcur2 = fcur2.astype(np.float64)
        
        # Calculate the cross-spectrum
        X = fref * (fcur.conj())

        # Smooth the amplitude spectrums
        dcur = np.sqrt(scipy.signal.convolve(fcur2,
                                            hanningwindow,
                                            "same"))

        dref = np.sqrt(scipy.signal.convolve(fref2,
                                        hanningwindow,
                                        "same"))
        
        # Smooth the cross-spectrum
        X = scipy.signal.convolve(X, hanningwindow, "same")
        
        dcs = np.abs(X)
        
        # Find useful frequencies
        freq_vec = sf.fftfreq(len(X) * 2, 1. / df)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= freq_min,freq_vec <= freq_max))
    
                                                 
        n = len(dcs)
        dcs = np.reshape(dcs, n)
        ds1 = np.reshape(dref, n)
        ds2 = np.reshape(dcur, n)
        
        # Create an array for coherency
        coh = np.zeros(n, np.complex128)
        
        # Find nonzero spectrum indexes
        valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
        valids = valids.T[0]
        
        # Compute coherency
        coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    
        coh[abs(coh) > 1.] = 1. + 0j
        
        # Compute mean coherency
        mcoh = np.mean(coh[index_range])
        use_coh = coh[index_range]
        
        # Cut coherencies that is close to 1 to avoid extremely large weight in linear regression
        use_coh[use_coh >= .99] = .99
            
        # Compute weights of phase shifts for linear regression
        w = 1.0 / (1.0 / (use_coh ** 2) - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)
        
        # Redefine frequencies to angle frequencies
        v = np.real(freq_vec[index_range]) * 2 * np.pi
        
        # Find phase shifts 
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]
       
        # Find a time shift in window
        m = np.sum(w*v*phi)/np.sum(w*(v**2)) 
        
        time_shift[i] = m
    
        # Find an uncertainty of estimate time-shift
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2) 

        time_shift_err[i] = e
        mean_coh[i] = np.real(mcoh)
        rec_time[i] = tmin+window_length/2.+i*window_step
        
       

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m
        


    return rec_time, time_shift, time_shift_err, mean_coh


def extract_dvv(rec_time, time_shift, time_shift_err, mean_coh, 
                r_time_type = "static", sides = "both", max_r_time = 80, 
                min_r_time = 15, velocity = np.nan, distance = np.nan, 
                max_dt = 0.5, min_coh = 0.5, max_err = 0.1):
    """
    Estimate relative velocity changes as the second part of multiple window cross-spectrum
    analysis of two waveform according to Clarke et al., 2011.

    Parameters
    ----------
    rec_time : float NumPy :class:`~numpy.ndarray`
        Record's time of centers of moving windows.
    time_shift : float NumPy :class:`~numpy.ndarray`
        Time shifts correspond to each window.
    time_shift_err : float NumPy :class:`~numpy.ndarray`
        Uncertainty of each time shift value.
    mean_coh : float NumPy :class:`~numpy.ndarray`
        Mean coherency in each window.
    r_time_type : str, optional
        If usefull record's times set manually, "static" should by used.
        If usefull record's times estimate based on velocity and distance.
        The default is "static".
    sides : str, optional
        tType of signal to be applied in stretching analisys. 
        If signal have two sides of records time "both" should be used.
        If singal have one side of recorde time "single" should be used.
        The default is "both".
    max_r_time : float, optional
        The maximum record time of used signals . The default is 80.
    min_r_time : float, optional
        The minimum record time of used signals.
        It should be setted if you use static r_time_type. The default is 15.
    velocity : float, optional
        DESCRIPTION. The default is np.nan.
    distance : float, optional
        Minimum velocity of waves to be removed in dvv estimation.
        It should be setted if you use dynamic r_time_type. The default is np.nan.
    max_dt : float, optional
        Maximum value of time shift to use in linear regression.
        It necessary to remove outlires.. The default is 0.5.
    min_coh : float, optional
        Minimum value of coherency between windowed signal.
        It necessary to remove measurments on higly unlikely windowed signals. The default is 0.5.
    max_err : float, optional
        Maximum value of uncertainty of time shift  between windowed signal.
        It necessary to remove low-quality measurments. The default is 0.1.

    Raises
    ------
    
        Raises warning if one of parameters was unacceptable setted.

    Returns
    -------
    m : float
        dv/v, Relative velocity changes between current and refrence waveforms.
    a : float
        Time shift between clocks on stations..
    em : float
        Uncertainty of dv/v estimation.
    ea : float
        Uncertainty of clock's shift estimation.

    """
    


    # Set usefull record's time
    if r_time_type == "static":
        lmlag = -min_r_time
        rmlag = min_r_time
    elif r_time_type == "dynamic":
        lmlag = -distance / velocity
        rmlag = distance / velocity
    else:
        raise "dtt_lag should be static or dynamic only"
    lMlag = -max_r_time
    rMlag = max_r_time
    
    # Create time indexes 
    if sides == "both":
        tindex = np.where(((rec_time >= lMlag) & (rec_time <= lmlag)) | ((rec_time >= rmlag) & (rec_time <= rMlag)))[0]
    elif sides == "left":
        tindex = np.where((rec_time >= lMlag) & (rec_time <= lmlag))[0]
    elif sides == "right":
        tindex = np.where((rec_time >= rmlag) & (rec_time <= rMlag))[0]
    else:
        raise "Sides must be right or left or both only"

    # Filter time shifts measurements according to thresholds
    cohindex = np.where(mean_coh >= min_coh)[0]
    errindex = np.where(time_shift_err <= max_err)[0]
    dtindex = np.where(np.abs(time_shift) <= max_dt)[0]
    index = np.intersect1d(cohindex, errindex)
    index = np.intersect1d(tindex, index)
    index = np.intersect1d(index, dtindex)
    
    errors = time_shift_err[index]
    X_filtered = rec_time[index]
    Value_filtered = time_shift[index]
    
    # Estimate weights for linear regression
    w = 1.0 / errors
    w[~np.isfinite(w)] = 1.0
    
    # Perform linear regression if there are enough points
    if len(Value_filtered) >= 2:
        m, a, em, ea = weighted_linear_regression(Value_filtered, X_filtered,w )
        return  m, a, em, ea
    else:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
@jit(nopython=True,  fastmath=True)
def weighted_linear_regression(dt_t, rec_t, weight):
    """
    Perfrom weighted linear regression according to Clarke et al., 2011

    Parameters
    ----------
    dt_t : float NumPy :class:`~numpy.ndarray`
        Useful time shifts correspond to each window..
    rec_t : float NumPy :class:`~numpy.ndarray`
        Useful record's time of centers of moving windows..
    weight : float NumPy :class:`~numpy.ndarray`
        Weights of each usefull time shift measurment.

    Returns
    -------
    m : float
        dv/v, Relative velocity changes between current and refrence waveforms.
    a : float
        Time shift between clocks on stations..
    em : float
        Uncertainty of dv/v estimation.
    ea : float
        Uncertainty of clock's shift estimation.

    """
 
    # Аштв еotal sum of weight
    w_sum = np.sum(weight)
    
    # Find weighted average of record_time
    t_aver = np.sum(weight*rec_t) / w_sum
    
    # Find weighted average of time shifts
    dt_average = np.sum(weight*dt_t) / w_sum
    
    # Find weighted average squaer of time shifts
    t2_aver = np.sum(weight*(dt_t**2)) / w_sum
    
    # Find line's inclination coefficient and respectively dv/v and its uncertainty
    m = -np.sum(weight*(rec_t - t_aver)*dt_t) / np.sum(weight*(rec_t-t_aver)**2)
    em = (1 / np.sum(weight*(rec_t-t_aver)**2))**0.5
    
    # Find record time's origin's shift and its uncertainty
    a = dt_average - m*t_aver
    ea = (t2_aver * em)**0.5
    
    return m, a, em, ea


def mwcs(second, first, freq_min, freq_max, df, tmin, window_length, window_step,
         hanningwindow,  r_time_type = "static", sides = "both", max_r_time = 80, 
         min_r_time = 15, velocity = np.nan, distance = np.nan,   max_dt = 0.5, 
         min_coh = 0.5,   max_err = 0.1):
    """
    Make multiple window cross-spectrum analysis of two waveform according to Clarke et al., 2011.
    

    Parameters
    ----------
    second : float NumPy :class:`~numpy.ndarray`
        Current waveform (size N) to be compared  with reference waveform.
    first : float NumPy :class:`~numpy.ndarray`
        reference waveform (size N) to be compared with current waveform.
    freq_min : float
        Minimum frequency of used signal's passband.
    freq_max : float
        maximum frequency of used signal's passband.
    df : float
        Sampling rate of used signals.
    tmin : float
        The leftmost record time of used signals.
    window_length : float
        Length of moving window to signals comparing.
        It is recommended to use windowlength at least 2 maximum periods to correct phase shift estimation at low periods.
    window_step : float
        Step of moving window to signals comparing.
        It is recommended to use window_step about 2 minimum periods to optimising perfomance time.
    hanningwindow : float NumPy :class:`~numpy.ndarray`
        Hanning smothing window (size M).
    r_time_type : str, optional
        If usefull record's times set manually, "static" should by used.
        If usefull record's times estimate based on velocity and distance.
        The default is "static".
    sides : str, optional
        tType of signal to be applied in stretching analisys. 
        If signal have two sides of records time "both" should be used.
        If singal have one side of recorde time "single" should be used.
        The default is "both".
    max_r_time : float, optional
        The maximum record time of used signals . The default is 80.
    min_r_time : float, optional
        The minimum record time of used signals.
        It should be setted if you use static r_time_type. The default is 15.
    velocity : float, optional
        DESCRIPTION. The default is np.nan.
    distance : float, optional
        Minimum velocity of waves to be removed in dvv estimation.
        It should be setted if you use dynamic r_time_type. The default is np.nan.
    max_dt : float, optional
        Maximum value of time shift to use in linear regression.
        It necessary to remove outlires.. The default is 0.5.
    min_coh : float, optional
        Minimum value of coherency between windowed signal.
        It necessary to remove measurments on higly unlikely windowed signals. The default is 0.5.
    max_err : float, optional
        Maximum value of uncertainty of time shift  between windowed signal.
        It necessary to remove low-quality measurments. The default is 0.1.
    Returns
    -------
    m : float
        dv/v, Relative velocity changes between current and refrence waveforms.
    a : float
        Time shift between clocks on stations..
    em : float
        Uncertainty of dv/v estimation.
    ea : float
        Uncertainty of clock's shift estimation.
    """
    

 
    
    # Make the first part of mwcs, that mean estimation of time shifts between windowed signals
    rec_time, time_shift, time_shift_err, mean_coh = extract_dtt(second, first, freq_min, freq_max, df,
                              tmin, window_length, window_step,hanningwindow)
    
    # Make the second part of mwcs, that mean dv/v estimation based on time evaluated time shifts
    m, a, em, ea = extract_dvv(rec_time, time_shift, time_shift_err, mean_coh, 
                               r_time_type, sides, max_r_time, 
                               min_r_time, velocity, distance, 
                               max_dt, min_coh, max_err)
    return m, a, em, ea


# To improve performance in case of processing several waveforms, vectorize the function defined above 
vectorize_mwcs = np.vectorize(mwcs, signature=("(n),(n),(),(),(),(),(),(k),(),(),(),(),(),(),(),(),()->(m)"))



def mwcs_for_combinations(waveform_gather, freq_min, freq_max, df, 
                          tmin, window_length, window_step,  smoothing_half_win=5,
                          r_time_type = "static", sides = "both", max_r_time = 80, 
                          min_r_time = 15, velocity = np.nan, distance = np.nan,   
                          max_dt = 0.5, min_coh = 0.5, max_err = 0.1):
    """
    Make MWCS analysis of all possible pairs of waveform.

    Parameters
    ----------
    waveform_gather : float NumPy :class:`~numpy.ndarray`
        Waveform gather (size (M,N)) to be compared  with each other.
    freq_min : float
        Minimum frequency of used signal's passband .
    freq_max : float
        Maximum frequency of used signal's passband .
    df : float
        Sampling rate of used signal's .
    tmin : float
        The leftmost record time of used signals .
    window_length : float
        length of moving window to signals comparing.
        It is recommended to use windowlength at least 2 maximum periods  to correct phase shift estimation at low periods.
    window_step : float
        tep of moving window to signals comparing.
        It is recommended to use window_step about 2 minimum periods to optimising perfomance time.
    smoothing_half_win : int, optional
        Half lenght of hann smoothing  window in samples. The default is 5.
    r_time_type : str, optional
        If usefull record's times set manually, "static" should by used.
        If usefull record's times estimate based on velocity and distance.
        The default is "static".
    sides : str, optional
        Type of signal to be applied in stretching analisys. 
        If signal have two sides of records time "both" should be used.
        If singal have one side of recorde time "single" should be used. 
        The default is "both".
    max_r_time : float, optional
        The maximum record time of used signals . The default is 80.
    min_r_time : float, optional
        The minimum record time of used signals.
        It should be setted if you use static r_time_type. The default is 15.
    velocity : float, optional
        DESCRIPTION. The default is np.nan.
    distance : float, optional
        Minimum velocity of waves to be removed in dvv estimation.
        It should be setted if you use dynamic r_time_type. The default is np.nan.
    max_dt : float, optional
        Maximum value of time shift to use in linear regression.
        It necessary to remove outlires.. The default is 0.5.
    min_coh : float, optional
        Minimum value of coherency between windowed signal.
        It necessary to remove measurments on higly unlikely windowed signals. The default is 0.5.
    max_err : float, optional
        Maximum value of uncertainty of time shift  between windowed signal.
        It necessary to remove low-quality measurments. The default is 0.1.

    Returns
    -------
    dvv : float NumPy :class:`~numpy.ndarray`
        Relative velocity changes between all pairs of waveforms.
    dvv_std : float NumPy :class:`~numpy.ndarray`
       Uncertainties of relative velocity changes between all pairs of waveforms.

    """
    
    # Get lenght of waveform gather    
    n = len(waveform_gather)
    
    # Make smothing hanning window 
    window_len = 2 * smoothing_half_win + 1
    import scipy.signal
    hanningwindow = scipy.signal.windows.hann(window_len).astype(np.float64)
    
    # Create an araay for analysis results
    result = np.zeros(( n*(n-1)//2, 4))
    
    # Define a progress bar
    bar = IncrementalBar('MWCS for all pairs of waveforms', max = n)
    
    #Perform MWCS analysis fo all possible pairs
    for i in range(n-1):

        # Make an array of current ith waveform's copies 
        firsts = np.vstack([waveform_gather[i] for ii in range(n-i-1)])
        
        # Make an array of waveform except for curren waveform and previous ones
        seconds = waveform_gather[i+1:].copy()
        
        # Perform analysis for all pairs with ith waveform
        result[i] = vectorize_mwcs(seconds, firsts, freq_min, freq_max, df, tmin, window_length, window_step,
                                   hanningwindow,  r_time_type, sides, max_r_time, min_r_time,
                                   velocity, distance,  max_dt, min_coh,  max_err)
        
        bar.next()
       

    bar.finish()
    
    # Factor dvv and its error by 100 to scale to %
    dvv = result.T[0]*100
    dvv_std = result.T[2]*100
    return dvv, dvv_std

