# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:15:35 2022

@author: BerezhnevYM
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from progress.bar import IncrementalBar

def waveform_stretch(ref_spline, current, t_idx, time_fac):
    """
    Stretch refernce waveform to some value and compared with current waveform.

    Parameters
    ----------
    ref_spline : spline SciPy: class `scipy.interpolate.UnivariateSpline`
        Spline object of reference waveform (size N) to be stretched and compared with current waveform.
    current : float NumPy :class:`~numpy.ndarray`
        Current waveform (size N) to be compared with stretched reference waveform.
    t_idx : int NumPy :class:`~numpy.ndarray`
        An array of record time indexes of unstretched reference waveform .
    time_fac : float
        Stretched factor to be applied to reference waveform.

    Returns
    -------
    cc_coef : float
        Correlation between reference waveform stretched with current time_fac and current waveform.

    """
  
    stretched_t_idx = t_idx*time_fac
    stretched_ref = ref_spline(stretched_t_idx)
    cc_coef = np.corrcoef(current, stretched_ref)[0, 1]
    return cc_coef

vec_waveform_stretch = np.vectorize(waveform_stretch, signature="(n),(n),(n),()->()")

def stretching(ref, cur, stretch_range, stretch_steps, 
               dtt_minlag, dtt_maxlag, freq_max, freq_min, sides="both"):
    """
    Make stretching analysis of two waveform according to Sens-Schönfelder and Wegler (2006).

    Parameters
    ----------
    ref : float NumPy :class:`~numpy.ndarray`
        Reference waveform (size N) to be stretched and compared with current waveform.
    cur : float NumPy :class:`~numpy.ndarray`
        Current waveform (size N) to be compared with stretched reference waveform.
    stretch_range : float
        Maximum amount of relative stretching and compressing value.
        Stretching will be applied in [-stretch_range, stretch_range] range.
    stretch_steps : int
        Number of steps in stretch range..
    dtt_minlag : float
        Minimum record time of signal to be used in analysis.
    dtt_maxlag : float
        Maximum record time of signal to be used in analysis..
    freq_max : float
        Maximum frequency of used signal's passband .
    freq_min : float
        Minimum frequency of used signal's passband .
    sides : str, optional
        Type of signal to be applied in stretching analisys. 
        If signal have two sides of records time "both" should be used.
        If singal have one side of recorde time "single" should be used. The default is "both".

    Returns
    -------
    dv : float
        Relative velocity change dv/v .
    error : float
        Errors in the dv/v measurements based on Weaver et al (2011).
    cc : float
        Correlation coefficient between the current and the best stretched reference waveform.
    cdp : float
        Correlation coefficient between the current and the initial reference waveform.

    """
  
    # Create an array of stretching factors
    stretchs = np.linspace(-stretch_range, stretch_range, stretch_steps)
    time_facs = np.exp(-stretchs)

    # Create a time axis
    if sides != 'single':
        time_idx = np.arange(len(ref)) - (len(ref) - 1.) / 2.
    else:
        time_idx = np.arange(len(ref))


    # Create a spline object for the reference trace
    ref_spline = UnivariateSpline(time_idx, ref, s=0)
    
    # Perform comperesing of stretched refrence wavefrom and current wavefrom
    cof = vec_waveform_stretch(ref_spline, cur, time_idx, time_facs)

    # Correlation coefficient between the original reference and current waveforms
    cdp = np.corrcoef(cur, ref)[0, 1]

    # Find maximum correlation coefficient of the refined  analysis
    cc = np.max(cof) 
    
    # Find the dv/v value corresponds to maximum correlation coefficient
    dv = cof[np.argmax(cof)]
    
    # Error computation according to Weaver et al (2011)
    T = 1 / (freq_max - freq_min)
    X = cc
    wc = np.pi * (freq_min + freq_max)
    t1 = dtt_minlag
    t2 = dtt_maxlag
    error = (np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, error, cc, cdp


# To improve performance in case of processing several waveforms, vectorize the function defined above
vectorize_stretch = np.vectorize(stretching, signature=("(n),(n),(),(),(),(),(),(),(),()->(m)"))

def stretching_for_combinations(waveform_gather,  stretch_range, stretch_steps, 
               dtt_minlag, dtt_maxlag, freq_max, freq_min, sides):
    """
    Make stretching analysis of all possible pairs of waveform.

    Parameters
    ----------
    waveform_gather : float NumPy :class:`~numpy.ndarray`
        Waveform gather (size (M,N)) to be stretched and compared with each other.
    stretch_range : float
        Maximum amount of relative stretching and compressing value.
        Stretching will be applied in [-stretch_range, stretch_range] range.
    stretch_steps : int
        Number of steps in stretch range..
    dtt_minlag : float
        Minimum record time of signal to be used in analysis.
    dtt_maxlag : float
        Maximum record time of signal to be used in analysis..
    freq_max : float
        Maximum frequency of used signal's passband .
    freq_min : float
        Minimum frequency of used signal's passband .
    sides : str, optional
        Type of signal to be applied in stretching analisys. 
        If signal have two sides of records time "both" should be used.
        If singal have one side of recorde time "single" should be used. The default is "both".
    Returns
    -------
    result : float NumPy :class:`~numpy.ndarray`
        First axis is relative velocity change dv/v.
        Second axis is  errors of the dv/v measurements based on Weaver et al (2011).

    """

    
    # Get lenght of waveform gather    
    n = len(waveform_gather)

    # Create an araay for analysis results
    result = np.zeros(( n*(n-1)//2, 4))
    
    # Define a progress bar
    bar = IncrementalBar('Stretching for all pairs of waveforms', max = n)
    
    #Perform stretching analysis fo all possible pairs
    for i in range(n-1):
        
        # Make an array of current ith waveform's copies 
        firsts = np.vstack([waveform_gather[i] for ii in range(n-i-1)])
        
        # Make an array of waveform except for curren waveform and previous ones
        seconds = waveform_gather[i+1:].copy()
        
        # Perform analysis for all pairs with ith waveform
        result[i] = vectorize_stretch(seconds, firsts,
                                        stretch_range, stretch_steps, 
                                        dtt_minlag, dtt_maxlag, freq_max, freq_min, sides)
        
        bar.next()
   
        
    bar.finish()
    
    # Factor dvv and its error by 100 to scale to %
    result[0] *= 100
    result[1] *= 100
    
    return result



