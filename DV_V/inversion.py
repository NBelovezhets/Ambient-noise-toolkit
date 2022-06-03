


import numpy as np
import itertools
from numba import jit
import matplotlib.pyplot as plt
from scipy import sparse 
import matplotlib.colors as mcolors
import seaborn as sns
import csr

plt.style.use("seaborn")
sns.set_style("whitegrid")

@jit(nopython=True)
def Cm_inv(n, beta):
    """
    Create an inverted priori covariance matrix (C_m) (size NxN) of model, where Cm_ij=exp(−|i−j|/2β)
    Parameters
    ----------
    n : int
        Number of model parameters.
    beta : float
        Characteristic correlation length of model parameters.

    Returns
    -------
    Cm_inv : float NumPy: class ~numpy.ndarray
        An inverted priori covariance matrix (size NxN) of model.

    """
    # Create an array for matrix
    Cm = np.zeros((n,n))
    # Set value for each element 
    for i, ivalue in enumerate(Cm):
        for j, jvalue in enumerate(ivalue):
            Cm[i,j] = np.exp((-1)*np.abs((i+1)-(j+1)) / (2 * beta)) 
    
    # Invert obteined matrix
    Cm_inv = np.linalg_inv(Cm)
    
    return Cm_inv

def Cd_inv(std):
    """
    Create inverted covariance matrix (size LxL) of data from measurment's uncertainties..

    Parameters
    ----------
    std : numpy float NumPy: class ~numpy.ndarray
        Vector of measured realtive velocity changes' uncertainties (size L).

    Returns
    -------
    float NumPy: class ~numpy.ndarray
        An inverted covariance matrix (size LxL) that describes  the  Gaussian  uncertainties of data.
        

    """
    
    return sparse.diags(1 / std**2)

def G_matrix(combinations, lenght):
    """
    Create a forward model matrix (size LxN) of problem according to Brenguier et al., 2014

    Parameters
    ----------
    combinations : numpy float NumPy: class ~numpy.ndarray
        An array (size L) of combinations of indexes pairs of used waveform
    lenght : int
        Number of observed waveform.

    Returns
    -------
    G : float SciPy: class ~scipy.sparse
        Matrix of forward model (size LxN) of problem.
    """
    
    dim_row = len(combinations)

    dim_cols = lenght
    
    # Make array of coordinates and values of non-zero elements of matrixes
    rows = np.hstack([[i, i]  for i,j in enumerate(combinations)])
    cols = np.hstack(combinations)
    value = np.hstack([1 if i%2 else -1 for i,j in enumerate(cols)])
    
    # Make a forward model matrix as a sparse matrix
    G = sparse.coo_matrix((value, (rows, cols)), shape=(dim_row,dim_cols), dtype=np.float64)
    
    return G
    
def plot_data_distribution(dvv):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.histplot(dvv,  kde=True)
    ax.set_xlabel(xlabel="dv/v, %", fontsize=20)
    ax.set_ylabel("Counts", fontsize=20)
    ax.tick_params(labelsize=18, width=2.5, length=10, color="black")
    plt.show()
    

def comb_list(lenght):
    """
    Create an array of combinations of pairs of waveform

    Parameters
    ----------
    lenght : int
        Number of waveforms.

    Returns
    -------
    float NumPy: class ~numpy.ndarray
        An array of combinations of waveform's pair indexes (size lenght*(lenght-1)/2 x 2).

    """
    return np.vstack(list(itertools.combinations(np.arange(0,lenght), 2)))




def Bayesian_inversion(dvv_filterd, G, Cd_inv,
                       Cm_inv, beta, alpha):
    """
    Make Bayesian least-square inversion of dvv measurments 
    according to Brenguier et al., 2014 and Gomez-Garcia et al., 2018
    

    Parameters
    ----------
    dvv_filterd : float NumPy: class ~numpy.ndarray
        Vector of measured realtive velocity changes between all pair of daily CCF (size L).
        
    G : float NumPy: class ~numpy.ndarray
        A forward model matrix (size LxN) of problem.
        
    Cd_inv : float NumPy: class ~numpy.ndarray
        An inverted covariance matrix (size LxL) that describes  the  Gaussian  uncertainties of data.
        
    Cm_inv : float NumPy: class ~numpy.ndarray
         An inverted priori covariance matrix (size NxN) of model.
        
    beta : float
         Characteristic correlation length between the daily variation in model to be obtained.
         
    alpha : float
        Weighting coefficient

    Returns
    -------
    m : float NumPy: class ~numpy.ndarray
        Model vector of daily velocity changes value (size N)
    m_std : TYPE
       Model vector of daily velocity changes uncertainties (size N).

    """
    
    G_T = G.T

    R = G_T.dot(Cd_inv)
    
    R = R.dot(G)
 
    R += Cm_inv * alpha

    R_inv = np.linalg.inv(R)
    
    m_std = np.diagonal(R_inv)**0.5
    
    R_tmp = G.dot(R_inv.T)  
    R = R_tmp.T

    del R_inv, Cm_inv, G_T
    
    R = (Cd_inv.T).dot(R_tmp)

    del Cd_inv
    
    R = R.T
    
    m = np.zeros_like(m_std)
    np.dot(R ,dvv_filterd, m)

    
    del R, G

    return m, m_std

def data_filter(dvv, dvv_std, cc_pair, cc_treshold=0.3):
    """
    Filter the data vector to avoid Nan value of measurments and its uncertainties and remove low-correlate daily combinations.
    

    Parameters
    ----------
    dvv : float NumPy: class ~numpy.ndarray
        Vector of measured realtive velocity changes between all pair of waveforms (size L).
    dvv_std : float NumPy: class ~numpy.ndarray
        Vector of measured realtive velocity changes uncertainties (size L).
    cc_pair : float NumPy: class ~numpy.ndarray
        Vector of correlation coefficient between waveform (size L).
    cc_treshold : float, optional
        Treshold of correlation coefficient to be removed. The default is 0.3.

    Returns
    -------
    indexes : float NumPy: class ~numpy.ndarray
        Vector of indexes that satisfied or not to requirements.

    """
    
    # Get not Nan values and its uncertainties
    non_nan_dvv_indexes = ~np.isnan(dvv)
    non_nan_std_indexes = ~np.isnan(dvv_std) 
    
    # Get only correlation coefficient that is above the treshold
    cc_treshold_indexes = cc_pair >= cc_treshold
    
    # Several logical and to find satisfied indexes
    index_dvv_std = np.logical_and(non_nan_dvv_indexes, non_nan_std_indexes)
    indexes = np.logical_and(index_dvv_std, cc_treshold_indexes)
    
    return indexes


def create_CCF_mask(cross_corr_mat, df, min_dt):
    """
    Create an mask for estimating correlation coefficient between CCF to save only informative part of signals.
    

    Parameters
    ----------
    cross_corr_mat : float NumPy: class ~numpy.ndarray
        CCF gather to be processed.
    df : float
        Sampling frequency of signals.
    min_dt : float
        Start point of important part of signal in seconds from the zero correlation-time.

    Returns
    -------
    mask : float NumPy: class ~numpy.ndarray
        Vector of indexes that contains informative part of signal or not.

    """
    n_counts = cross_corr_mat.shape[1]
    
    center_index = np.floor((n_counts - 1.) / 2.)
    left_part = np.arange(0, center_index - df*min_dt + 1)
    right_part = np.arange(center_index + df*min_dt, n_counts)
    uses_indexes = np.hstack([left_part, right_part]).astype(int)
    mask = np.zeros((n_counts))
    mask[uses_indexes] = 1
    
    return mask

def correlation_coefficient_matrix(cross_corr_mat, df, dt_minlag):
    """
    Create correlation coefficient matrix of used waveform

    Parameters
    ----------
    cross_corr_mat : float NumPy: class ~numpy.ndarray
        CCF gather to be processed.
    df : float
        Sampling frequency of signals.
    min_dt : float
         Start point of important part of signal in seconds from the zero correlation-time.


    Returns
    -------
    cc_matrix : float NumPy: class ~numpy.ndarray
        Correlation coefficient matrix (size NxN) between used waveforms
    cc_array : float NumPy: class ~numpy.ndarray
        Vector of correlation coefficient (size N x (N-1)/2) between used waveforms.

    """
    mat = cross_corr_mat.copy()
    n = mat.shape[0]
    
    # Create an array of combinations
    combinations = comb_list(n)
   
    cross_corr_number = cross_corr_mat.shape[0]
    combinations_number = len(combinations)
    
    # Create an array to save processing results
    cc_matrix = np.zeros((cross_corr_number, cross_corr_number))
    cc_array = np.zeros((combinations_number))
    
    # Reject unstable central part of CCF before estimate corrrelation coefficient
    mask = create_CCF_mask(mat, df, dt_minlag)
    mat *=mask
    
    # Compute corrrelation coefficients
    for i, i_comb in enumerate(combinations):
        n, k = i_comb
        cc_coeff = np.corrcoef(mat[n], mat[k])[0][1]
        
        cc_matrix[n, k] = cc_coeff
        cc_matrix[k, n] = cc_coeff
        cc_array[i] = cc_coeff
        
    for i in range(cross_corr_number):
        cc_matrix[i, i] = 1
    
    return cc_matrix, cc_array

def plot_cc_matrix(cc_matrix, array_of_date):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    
    
    X, Y = np.meshgrid(array_of_date, array_of_date)
    
    normalize = mcolors.Normalize(vmin=0.3, vmax=1)
    colors = [(1, 1, 224/255), (1, 1, 0), (1, 0., 0.), (0, 0, 0)] 
    cmap = mcolors.LinearSegmentedColormap.from_list('test', colors, N=7)
    
    pcolor = ax.pcolormesh(X, Y, cc_matrix, cmap=cmap, norm=normalize)
    
    ax.set_xlabel("Date", fontsize=20)
    ax.set_ylabel("Date", fontsize=20)
    ax.set_title("CC matrix", fontsize=22)
    ax.tick_params(axis='y', labelsize=18, length=10, color="black")
    ax.tick_params(axis='x', rotation=90, labelsize=18, length=10, color="black")
    ax.invert_yaxis()
    
    cb = fig.colorbar(pcolor, extend='both')
    cb.ax.set_ylabel("Correlation", fontsize=20)
    cb.outline.set_linewidth(2)
    
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(18)
    plt.show()

def plot_dvv_curve(dvv, std, array_of_date, long_term_dvv=None):
    
    fig, ax = plt.subplots(1,1, figsize=(16, 4))

    
    if long_term_dvv is None:
        ax.plot(array_of_date, dvv, "o-", color="black", label="dv/v")
        ax.fill_between(array_of_date, (dvv - std), (dvv + std), color="grey", alpha=0.4, label="dv/v uncertanty")
    else:
        ax.plot(array_of_date, dvv, "o-", color="black", label="short-term dv/v")
        ax.plot(array_of_date,long_term_dvv,  "-", color="blue", label="long-term dv/v")

    ax.set_ylabel(ylabel="dv/v, %", fontsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='x', which='major', labelsize=14, labelrotation=-90)
    ax.grid(True)
    ax.set_xlabel("Date", fontsize=18)
    ax.legend(fontsize=14)
    plt.show()


