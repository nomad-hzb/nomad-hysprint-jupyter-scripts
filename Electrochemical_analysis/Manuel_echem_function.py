
color_cm = ["#004f84ff", "#6cabe9ff", "#d15e57ff", "#ff7f2aff","#808080ff","#8787deff"]
# Expanded with 5 additional colors that harmonize with the original palette
color_cm = color_cm + ["#00a591ff", "#b2df8aff", "#f9c74fff"]

## Impedance functions # 
# 
# !pip install impedance -> Run in case the module is not installed
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
import matplotlib.pyplot as plt
from impedance.visualization import plot_nyquist

def impedance_fit2(frequencies,Z,circuit,initial_guess,constants):
    
    frequencies, Z = preprocessing.ignoreBelowX(frequencies,Z)
    
    circuit = CustomCircuit(circuit, initial_guess=initial_guess,constants=constants)
        
    circuit.fit(frequencies, Z)
    Z_fit = circuit.predict(frequencies)
    return Z_fit, circuit

def impedance_fit1(frequencies,Z,circuit,initial_guess):
    
    frequencies, Z = preprocessing.ignoreBelowX(frequencies,Z)
    
    circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        
    circuit.fit(frequencies, Z)
    Z_fit = circuit.predict(frequencies)
    return Z_fit, circuit


def coverage_EIS(Rct_ITO,Rct_SAM):
    theta=1-Rct_ITO/Rct_SAM
    return theta

def impedance_analysis(df,experiment,axs,aa=0, plot=True, fit=True, color=color_cm[0],name=None):
    ## Filter the dataframe for the frequency column
    df_EIS = df[df.iloc[:, 10].astype(str).str.strip() != '']

    k=experiment.count("EIS")
    clusters=split_by_k_largest_gaps(df_EIS, k, x_col="Time (s)")

    #df_EIS = df_EIS[:50]    # Limit the dataset to check the convergence
    df_EIS=clusters[aa]
    ## Prepare data
    frequencies=np.double(df_EIS.iloc[10:,5])
    Z_r=np.double(df_EIS.iloc[10:,10])
    Z_i=np.double(df_EIS.iloc[10:,11])
    Z = Z_r + 1j * Z_i  # Combine real and imaginary parts into complex impedance

    Rct=[]
    i=0

    ## Plotting
    if plot == True:
        axs.plot(Z_r,-Z_i, "o",color=color,label=name)
        axs.set_aspect('equal')
        axs.set_xlabel(r"$Z'$")
        axs.set_ylabel(r"$Z''$")   
        plt.tight_layout()

     #axs[1].plot(frequencies**(-1/2),Z_r, "o",color=color_cm[i],label=sample_name[i])

    if fit == True:
        if np.max(Z_r)<100000000:
            circuitA = 'R0-p(R2-W1,CPE1)'
            initial_guess = [None, 8, 500, 1e-5,0.5]
            constants={"R0":16.5}
            Z_fit,circuit=impedance_fit2(frequencies,Z,circuitA,initial_guess,constants)
            print(circuit.parameters_)
            Rct.append(circuit.parameters_[0])
            #wo.append(circuit.parameters_[1])
            if i==1:
                saveW=circuit
        else:
            circuitA='R0-p(R1,p(R2-Wo1,CPE1))'
            initial_guess=[None, 5000, 5000, 500, 2, 1e-6,0.5]  # R0, R1, C1
            constants={"R0":16.5}
            Z_fit,circuit=impedance_fit2(frequencies,Z,circuitA,initial_guess,constants)
            print(circuit.parameters_)
            Rct.append(circuit.parameters_[1])

        # Get sorting indices based on frequency

        sorted_indices = np.argsort(frequencies)

        # Sort both arrays
        Z_fit = Z_fit[sorted_indices]
        if plot == True:
            axs.plot(np.real(Z_fit),-np.imag(Z_fit), "-",color=color)

    
    return df_EIS,Rct


## DPV functions --------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import GaussianModel

def gaussian(x, amp, cen, wid):
    """1D Gaussian: amp = area, cen = center, wid = standard deviation."""
    return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def fit_gaussian(x, y, amp_init=1, cen_init=0, wid_init=1):
    """
    Fit a Gaussian to data (x, y) using lmfit.
    Returns the lmfit ModelResult object.
    """
    model = GaussianModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)
    return result

def DPV_analysis(df,experiment,axs, area=1,aa=0, plot=True, fit=True, color=color_cm[0],name=None):

    df_exp = df[df[" Reverse I (A)"].astype(str).str.strip() != '']   # Filter the data based on the column Reverse I

    k=experiment.count("DPV")
    clusters=split_by_k_largest_gaps(df_exp, k, x_col="Time (s)")

    df_exp=clusters[aa]

    exp_time=np.double(df_exp["Time (s)"])
    exp_voltage=np.double(df_exp[" Voltage (V)"])
    exp_current=(np.double(df_exp[" Current (A)"])-np.double(df_exp[" Reverse I (A)"]))

    df_exp["DPV current"]=exp_current

    ip_max=[]
    HOMO_max=[]
    center_E=[]
    FWHM_E=[]
    fit_variables=[]

    if plot == True: 
        
        # fig,axs=plt.subplots(figsize=(5,5))
        axs.axhline(y=0,c="grey")
        axs.plot(exp_voltage,exp_current*1e3/area, "-",color=color,label=name)

        axs.set_xlabel("Voltage (V vs Ag/AgCl)")
        axs.set_ylabel("Differential Current (mA/cm$^2$)")
        plt.tight_layout()

    if fit == True:

        # fig,axs=plt.subplots(figsize=(5,5))

        ##---- Baseline fit
        # Substract a linear baseline 
        maskB = (exp_voltage >= -0.2) & (exp_voltage <= 0)
        coeffs = np.polyfit(exp_voltage[maskB], exp_current[maskB], 1)
        m, b = coeffs
        baseline = m * exp_voltage + b

        exp_current_base=(exp_current-baseline)*1e3

        ##---- Gaussianns fit
        mask = (exp_voltage >= 0) & (exp_voltage <= 0.35)

        # Apply the mask
        filtered_voltage = exp_voltage[mask]
        filtered_current = exp_current_base[mask]
        result = fit_gaussian(filtered_voltage, filtered_current)
        #result = fit_pseudovoigt(filtered_voltage, filtered_current)
        # Define a new x-range (e.g., for extrapolation or visualization)
        x_new = np.linspace(0, 0.8, 500)
        baseline_new=m * x_new + b
        y_new = result.model.eval(params=result.params, x=x_new)

        ip_max.append(result.params['height'].value)
        center_E.append(result.params['center'].value)
        FWHM_E.append(result.params['fwhm'].value)

        if plot==True:
            axs.plot(exp_voltage,baseline,color=color,alpha=0.5) 
            axs.plot(x_new, y_new, '--', color="grey" )       
            # axs.plot(exp_voltage,exp_current*1e6/area, "-")

        fit_variables=[ip_max,center_E,FWHM_E]
    return df_exp,fit_variables

# df_exp=DPV_analysis(df,plot=True,fit=True)

### Cylcic voltammetry analysis

def CV_analysis(df,reference_values,threshold,plot_SS=False):

    df_exp = df[df[" Reverse I (A)"].astype(str).str.strip() == '']   # Filter the data based on the column Reverse I
    df_CV = df_exp[df_exp.iloc[:, 10].astype(str).str.strip() == '']
    
    Voltage=df_CV[" Voltage (V)"]
    Current=df_CV[" Current (A)"]
    Time=df_CV["Time (s)"]

    # Calculate scan speed
    scan_speed=np.abs(np.diff(Voltage)/np.diff(Time))
    window_size = 10
    kernel = np.ones(window_size) / window_size  # averaging kernel
    scan_speed_smooth = np.convolve(scan_speed, kernel, mode='same')

    if plot_SS == True:
        plt.plot(Time[:-1],scan_speed_smooth)
        #plt.plot(Voltage,Current)
        plt.xlabel("Time(s)")
        plt.ylabel("Scan_speed")

    df_CV["scan_speed"] = np.append(scan_speed_smooth,1)

    # Reference values and threshold
    #reference_values = [0.05, 0.1, 0.2, 0.3, 0.5 ]
    #threshold = 0.02

    # Define bins around each reference value
    bins = []
    labels = []

    for val in reference_values:
        bins.append((val - threshold, val + threshold))
        labels.append(f"{val:.2f}")

    # Function to assign bin label
    def assign_bin(x):
        for (low, high), label in zip(bins, labels):
            if low <= x <= high:
                return label
        return np.nan  # if it doesn't fit any bin

    # Apply to the DataFrame
    df_CV['Speed_bin'] = df_CV["scan_speed"].apply(assign_bin)

    return df_CV

def assign_cycles(df1):
    from scipy.signal import find_peaks        
    # --- Find maxima ---
    max_idx, _ = find_peaks(df1[" Voltage (V)"])
    min_idx, _ = find_peaks(-df1[" Voltage (V)"])

    df1["Cycle"] = np.nan

    for i in range(len(min_idx) - 1):
        start, end = min_idx[i], min_idx[i + 1]
        df1["Cycle"].iloc[start:end] = i + 1  # Cycle 1, 2, 3, ...

    return df1


def CV_scan_speed(df_CV,axs=None,scan_speed='0.05',plot=True,plot_cycle=1,area=1,color=color_cm[0],name=None):
    df1 = df_CV[df_CV['Speed_bin'] == scan_speed]

    df1=assign_cycles(df1)

    if plot == True:
        if plot_cycle=='all':
            # plt.figure()
            alpha=1
            for cycle, df_cycle in df1.groupby("Cycle"):
                axs.plot(df_cycle[" Voltage (V)"],df_cycle[" Current (A)"]*1e3,color=color,alpha=alpha)
                alpha*=0.9
            axs.legend()
        else:
            # plt.figure()
            df_cycle2 = df1[df1["Cycle"] == plot_cycle]
            axs.plot(df_cycle2[" Voltage (V)"], df_cycle2[" Current (A)"]*1e3, label=name,color=color)
            axs.legend()
        axs.set_xlabel("Voltage (V vs Ag/AgCl)")
        axs.set_ylabel("Current (mA/cm$^2$)")
        
    return df1
#df1.loc[min_idx[-1]:, "Cycle"] = len(min_idx)

import numpy as np
import pandas as pd

def split_by_k_largest_gaps(df, k, x_col="x"):
    if k <= 1 or df.empty:
        return [df.copy()]

    work = df.sort_values(x_col).reset_index(drop=True)
    x = work[x_col].to_numpy()
    if len(x) <= k:
        # each point is its own cluster at most
        return [work.iloc[[i]] for i in range(len(x))]

    diffs = np.diff(x)                           # gaps between consecutive x
    # indices of the (k-1) largest gaps
    cut_idx = np.argpartition(diffs, -(k-1))[-(k-1):]
    cut_idx = np.sort(cut_idx)                   # split after these indices

    starts = np.r_[0, cut_idx + 1]
    ends   = np.r_[cut_idx + 1, len(work)]
    clusters = [work.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, ends)]
    return clusters

def get_experiment_before_cv(experiments):
    """
    Returns the experiment name that appears just before the first 'CV' in the list.
    If 'CV' is not found or it's the first element, returns None.
    """
    try:
        idx = experiments.index("CV")
        if idx > 0:
            return experiments[idx - 1]
        else:
            return None  # 'CV' is first, nothing before it
    except ValueError:
        return None  # 'CV' not in list
    


def CV_analysis_v2(df,experiments,reference_values,threshold,plot_SS=False):


    df_exp = df[df[" Reverse I (A)"].astype(str).str.strip() == '']   # Filter the data based on the column Reverse I
    df_CV = df_exp
    
    Voltage=df_CV[" Voltage (V)"]
    Current=df_CV[" Current (A)"]
    Time=df_CV["Time (s)"]

    # Calculate scan speed
    scan_speed=np.abs(np.diff(Voltage)/np.diff(Time))
    window_size = 10
    kernel = np.ones(window_size) / window_size  # averaging kernel
    scan_speed_smooth = np.convolve(scan_speed, kernel, mode='same')

    if plot_SS == True:
        plt.plot(Time[:-1],scan_speed_smooth)
        #plt.plot(Voltage,Current)
        plt.xlabel("Time(s)")
        plt.ylabel("Scan_speed")

    df_CV["scan_speed"] = np.append(scan_speed_smooth,1)

    # Reference values and threshold
    #reference_values = [0.05, 0.1, 0.2, 0.3, 0.5 ]
    #threshold = 0.02

    # Define bins around each reference value
    bins = []
    labels = []

    for val in reference_values:
        bins.append((val - threshold, val + threshold))
        labels.append(f"{val:.2f}")

    # Function to assign bin label
    def assign_bin(x):
        for (low, high), label in zip(bins, labels):
            if low <= x <= high:
                return label
        return np.nan  # if it doesn't fit any bin

    # Apply to the DataFrame
    df_CV['Speed_bin'] = df_CV["scan_speed"].apply(assign_bin)

    return df_CV

def assign_cycles_NOMAD(df1):
    from scipy.signal import find_peaks        
    # --- Find maxima ---
    max_idx, _ = find_peaks(df1["voltage"])
    min_idx, _ = find_peaks(-df1["voltage"])

    df1["Cycle"] = np.nan

    for i in range(len(min_idx) - 1):
        start, end = min_idx[i], min_idx[i + 1]
        df1["Cycle"].iloc[start:end] = i + 1  # Cycle 1, 2, 3, ...

    return df1


def CV_analysis_NOMAD(Time,Voltage,Current,reference_values,threshold,plot_SS=False):

    df_CV = pd.DataFrame({"time": Time, "voltage": Voltage, "current": Current})   

    scan_speed=np.abs(np.diff(Voltage)/np.diff(Time))
    window_size = 10
    kernel = np.ones(window_size) / window_size  # averaging kernel
    scan_speed_smooth = np.convolve(scan_speed, kernel, mode='same')

    if plot_SS == True:
        plt.plot(Time[:-1],scan_speed_smooth)
        #plt.plot(Voltage,Current)
        plt.xlabel("Time(s)")
        plt.ylabel("Scan_speed")

    df_CV["scan_speed"] = np.append(scan_speed_smooth,1)

    # Reference values and threshold
    #reference_values = [0.05, 0.1, 0.2, 0.3, 0.5 ]
    #threshold = 0.02

    # Define bins around each reference value
    bins = []
    labels = []

    for val in reference_values:
        bins.append((val - threshold, val + threshold))
        labels.append(f"{val:.2f}")

    # Function to assign bin label
    def assign_bin(x):
        for (low, high), label in zip(bins, labels):
            if low <= x <= high:
                return label
        return np.nan  # if it doesn't fit any bin

    # Apply to the DataFrame
    df_CV['Speed_bin'] = df_CV["scan_speed"].apply(assign_bin)

    return df_CV

    
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    q = np.ones(L)
    for k in range(niter):
        Q = sparse.spdiags(q, 0, L, L)
        Z = Q + lam * D.dot(D.transpose())
        z = spsolve(Z, q*y)
        q = p * (y > z) + (1-p) * (y < z)
    return z

