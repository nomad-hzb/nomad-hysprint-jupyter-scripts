# fitting_engine.py
"""
Consolidated fitting module
Contains peak detection, model definitions, and fitting orchestration
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.optimize import curve_fit
from lmfit import Model, Parameters, CompositeModel
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel, LinearModel, PolynomialModel, ExponentialModel
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import config
from utils import debug_print


# =============================================================================
# PEAK DETECTION
# =============================================================================

class PeakDetector:
    """Class to handle automatic peak detection in photoluminescence spectra"""
    
    def __init__(self):
        self.default_params = {
            'height': None,
            'threshold': None,
            'distance': 5,
            'prominence': None,
            'width': None,
            'wlen': None,
            'rel_height': 0.5,
            'plateau_size': None
        }
        
    def detect_peaks(self, wavelengths, intensities, **kwargs):
        """
        Detect peaks in a spectrum
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
        **kwargs : dict
            Peak detection parameters
            
        Returns:
        --------
        list: List of detected peak information
        """
        # Merge default parameters with user input
        params = {**self.default_params, **kwargs}
        
        # Find peaks
        peaks, properties = find_peaks(intensities, **params)
        
        # Calculate additional peak properties
        peak_info = []
        
        for i, peak_idx in enumerate(peaks):
            # Basic peak information
            peak_data = {
                'index': peak_idx,
                'center': wavelengths[peak_idx],
                'height': intensities[peak_idx],
                'prominence': properties.get('prominences', [None])[i] if 'prominences' in properties else None
            }
            
            # Calculate peak width
            try:
                widths, width_heights, left_ips, right_ips = peak_widths(
                    intensities, [peak_idx], rel_height=params['rel_height']
                )
                peak_data['width'] = widths[0]
                peak_data['width_height'] = width_heights[0]
                peak_data['left_base'] = wavelengths[int(left_ips[0])] if left_ips[0] < len(wavelengths) else wavelengths[0]
                peak_data['right_base'] = wavelengths[int(right_ips[0])] if right_ips[0] < len(wavelengths) else wavelengths[-1]
                
                # Estimate sigma for Gaussian approximation
                # FWHM = 2.355 * sigma for Gaussian
                peak_data['sigma'] = (peak_data['right_base'] - peak_data['left_base']) / 2.355
                
            except Exception as e:
                # Fallback values
                peak_data['width'] = 10.0
                peak_data['sigma'] = 5.0
                peak_data['width_height'] = peak_data['height'] / 2
                
            # Calculate peak area (rough approximation)
            try:
                left_idx = max(0, peak_idx - int(peak_data['width']))
                right_idx = min(len(intensities), peak_idx + int(peak_data['width']))
                peak_data['area'] = np.trapz(
                    intensities[left_idx:right_idx],
                    wavelengths[left_idx:right_idx]
                )
            except:
                peak_data['area'] = peak_data['height'] * peak_data['sigma'] * np.sqrt(2 * np.pi)
                
            peak_info.append(peak_data)
            
        return peak_info
        
    def detect_peaks_advanced(self, wavelengths, intensities, min_height=None, min_prominence=None, 
                             min_distance=5, adaptive_threshold=True):
        """
        Advanced peak detection with adaptive parameters
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
        min_height : float, optional
            Minimum peak height
        min_prominence : float, optional
            Minimum peak prominence
        min_distance : int
            Minimum distance between peaks
        adaptive_threshold : bool
            Use adaptive thresholding based on signal statistics
            
        Returns:
        --------
        list: List of detected peak information
        """
        # Calculate signal statistics for adaptive thresholding
        if adaptive_threshold:
            signal_mean = np.mean(intensities)
            signal_std = np.std(intensities)
            signal_max = np.max(intensities)
            
            # Adaptive parameters
            if min_height is None:
                min_height = signal_mean + 2 * signal_std
            if min_prominence is None:
                min_prominence = signal_std
                
        # Smooth the signal for better peak detection
        smoothed_intensities = self._smooth_signal(intensities)
        
        # Detect peaks on smoothed signal
        peaks, properties = find_peaks(
            smoothed_intensities,
            height=min_height,
            prominence=min_prominence,
            distance=min_distance
        )
        
        # Refine peak positions on original signal
        refined_peaks = []
        for peak_idx in peaks:
            # Search for maximum in neighborhood
            search_range = 3
            start_idx = max(0, peak_idx - search_range)
            end_idx = min(len(intensities), peak_idx + search_range + 1)
            
            local_max_idx = np.argmax(intensities[start_idx:end_idx])
            refined_peak_idx = start_idx + local_max_idx
            refined_peaks.append(refined_peak_idx)
            
        # Calculate detailed peak properties
        peak_info = []
        for i, peak_idx in enumerate(refined_peaks):
            peak_data = self._calculate_peak_properties(
                wavelengths, intensities, peak_idx
            )
            peak_info.append(peak_data)
            
        return peak_info
        
    def _smooth_signal(self, signal, window_size=5):
        """Apply smoothing to signal for better peak detection"""
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(signal, size=window_size, mode='reflect')
        
    def _calculate_peak_properties(self, wavelengths, intensities, peak_idx):
        """Calculate comprehensive peak properties"""
        peak_data = {
            'index': peak_idx,
            'center': wavelengths[peak_idx],
            'height': intensities[peak_idx]
        }
        
        # Find peak base points
        left_base_idx, right_base_idx = self._find_peak_base(intensities, peak_idx)
        
        # Calculate width and sigma
        width_nm = wavelengths[right_base_idx] - wavelengths[left_base_idx]
        peak_data['width'] = width_nm
        peak_data['sigma'] = width_nm / 2.355  # Gaussian approximation
        
        # Calculate area
        peak_data['area'] = np.trapz(
            intensities[left_base_idx:right_base_idx+1],
            wavelengths[left_base_idx:right_base_idx+1]
        )
        
        # Calculate prominence
        peak_data['prominence'] = self._calculate_prominence(intensities, peak_idx)
        
        # Fit local Gaussian for better parameter estimation
        try:
            fit_params = self._fit_local_gaussian(
                wavelengths, intensities, peak_idx, left_base_idx, right_base_idx
            )
            peak_data.update(fit_params)
        except:
            pass
            
        return peak_data
        
    def _find_peak_base(self, intensities, peak_idx):
        """Find the base points of a peak"""
        # Search left for minimum
        left_idx = peak_idx
        while left_idx > 0 and intensities[left_idx-1] < intensities[left_idx]:
            left_idx -= 1
            
        # Search right for minimum
        right_idx = peak_idx
        while right_idx < len(intensities)-1 and intensities[right_idx+1] < intensities[right_idx]:
            right_idx += 1
            
        return left_idx, right_idx
        
    def _calculate_prominence(self, intensities, peak_idx):
        """Calculate peak prominence"""
        try:
            prominences = peak_prominences(intensities, [peak_idx])
            return prominences[0][0]
        except:
            return None
            
    def _fit_local_gaussian(self, wavelengths, intensities, peak_idx, left_idx, right_idx):
        """Fit a Gaussian to local peak region"""
        # Extract local region
        local_x = wavelengths[left_idx:right_idx+1]
        local_y = intensities[left_idx:right_idx+1]
        
        # Initial guess
        center_guess = wavelengths[peak_idx]
        height_guess = intensities[peak_idx]
        sigma_guess = (wavelengths[right_idx] - wavelengths[left_idx]) / 4
        
        # Gaussian function
        def gaussian(x, center, height, sigma, offset):
            return height * np.exp(-((x - center) / sigma)**2) + offset
            
        # Fit
        popt, _ = curve_fit(
            gaussian,
            local_x,
            local_y,
            p0=[center_guess, height_guess, sigma_guess, np.min(local_y)],
            maxfev=1000
        )
        
        return {
            'fitted_center': popt[0],
            'fitted_height': popt[1],
            'fitted_sigma': popt[2],
            'fitted_offset': popt[3]
        }
        
    def filter_peaks(self, peak_info, min_height=None, min_prominence=None, 
                    min_width=None, max_width=None, wavelength_range=None):
        """
        Filter detected peaks based on criteria
        
        Parameters:
        -----------
        peak_info : list
            List of peak information dictionaries
        min_height : float, optional
            Minimum peak height
        min_prominence : float, optional
            Minimum peak prominence
        min_width : float, optional
            Minimum peak width
        max_width : float, optional
            Maximum peak width
        wavelength_range : tuple, optional
            (min_wavelength, max_wavelength) range
            
        Returns:
        --------
        list: Filtered peak information
        """
        filtered_peaks = []
        
        for peak in peak_info:
            # Height filter
            if min_height is not None and peak['height'] < min_height:
                continue
                
            # Prominence filter
            if min_prominence is not None and peak.get('prominence', 0) < min_prominence:
                continue
                
            # Width filters
            if min_width is not None and peak.get('width', 0) < min_width:
                continue
            if max_width is not None and peak.get('width', float('inf')) > max_width:
                continue
                
            # Wavelength range filter
            if wavelength_range is not None:
                min_wl, max_wl = wavelength_range
                if peak['center'] < min_wl or peak['center'] > max_wl:
                    continue
                    
            filtered_peaks.append(peak)
            
        return filtered_peaks
        
    def merge_close_peaks(self, peak_info, distance_threshold=10):
        """
        Merge peaks that are too close together
        
        Parameters:
        -----------
        peak_info : list
            List of peak information dictionaries
        distance_threshold : float
            Minimum distance between peaks (in nm)
            
        Returns:
        --------
        list: Merged peak information
        """
        if len(peak_info) <= 1:
            return peak_info
            
        # Sort peaks by center wavelength
        sorted_peaks = sorted(peak_info, key=lambda x: x['center'])
        
        merged_peaks = []
        current_peak = sorted_peaks[0]
        
        for next_peak in sorted_peaks[1:]:
            distance = abs(next_peak['center'] - current_peak['center'])
            
            if distance < distance_threshold:
                # Merge peaks - keep the one with higher prominence
                if next_peak.get('prominence', 0) > current_peak.get('prominence', 0):
                    current_peak = next_peak
            else:
                merged_peaks.append(current_peak)
                current_peak = next_peak
                
        merged_peaks.append(current_peak)
        
        return merged_peaks


# =============================================================================
# FITTING MODELS
# =============================================================================

class FittingModels:
    """Class to handle fitting of photoluminescence spectra"""
    
    def __init__(self):
        self.available_peak_models = {
            'Gaussian': GaussianModel,
            'Voigt': VoigtModel,
            'Lorentzian': LorentzianModel,
            'Linear': LinearModel,
            'Polynomial': PolynomialModel
        }
        
        self.available_background_models = {
            'Linear': LinearModel,
            'Polynomial': PolynomialModel,
            'Exponential': ExponentialModel
        }
        
    def create_composite_model(self, fit_params):
        """
        Create a composite model based on the fitting parameters
        
        Parameters:
        -----------
        fit_params : dict
            Dictionary containing model parameters
            
        Returns:
        --------
        CompositeModel: The composite model for fitting
        Parameters: Initial parameter values
        """
        model = None
        params = Parameters()
        
        # Add background model
        if fit_params['background_model'] != 'None':
            if fit_params['background_model'] == 'Polynomial':
                bg_model = PolynomialModel(degree=fit_params['poly_degree'], prefix='bg_')
            elif fit_params['background_model'] == 'Linear':
                bg_model = LinearModel(prefix='bg_')
            elif fit_params['background_model'] == 'Exponential':
                bg_model = ExponentialModel(prefix='bg_')
            else:
                bg_model = LinearModel(prefix='bg_')  # Default fallback
                
            model = bg_model
            params.update(bg_model.make_params())
            
        # Add peak models
        # Add peak models
        for i, peak_info in enumerate(fit_params['peak_models']):
            peak_type = peak_info['type']
            
            # Special handling for Polynomial which needs degree parameter
            if peak_type == 'Polynomial':
                degree = peak_info.get('poly_degree', 2)
                peak_model = PolynomialModel(degree=degree, prefix=f'p{i}_')
            else:
                peak_model_class = self.available_peak_models[peak_type]
                peak_model = peak_model_class(prefix=f'p{i}_')
            
            if model is None:
                model = peak_model
            else:
                model = model + peak_model
                
            # Set initial parameters
            peak_params = peak_model.make_params()
            
            # Set initial values based on UI input
            if peak_info['type'] == 'Gaussian':
                peak_params[f'p{i}_center'].set(value=peak_info['center'], min=peak_info['center']-50, max=peak_info['center']+50)
                peak_params[f'p{i}_amplitude'].set(value=peak_info['height']*peak_info['sigma']*np.sqrt(2*np.pi), min=0)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=0.001, max=100)
            elif peak_info['type'] == 'Polynomial':
                # Polynomial - use fitted coefficients if available
                degree = peak_info.get('poly_degree', 2)
                fitted_coeffs = peak_info.get('fitted_coeffs', None)
                
                if fitted_coeffs and len(fitted_coeffs) >= degree + 1:
                    # Use previously fitted coefficients
                    for j in range(degree + 1):
                        peak_params[f'p{i}_c{j}'].set(value=fitted_coeffs[j])
                else:
                    # Start from zero and let fitter find values
                    for j in range(degree + 1):
                        peak_params[f'p{i}_c{j}'].set(value=0.0, min=-1000, max=1000)
            elif peak_info['type'] == 'Linear':
                # Linear - initialize slope and intercept
                peak_params[f'p{i}_slope'].set(value=0.0)
                peak_params[f'p{i}_intercept'].set(value=0.0)
            elif peak_info['type'] == 'Lorentzian':
                peak_params[f'p{i}_center'].set(value=peak_info['center'], min=peak_info['center']-50, max=peak_info['center']+50)
                peak_params[f'p{i}_amplitude'].set(value=peak_info['height']*peak_info['sigma']*np.pi, min=0)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=0.001, max=100)
            elif peak_info['type'] == 'Voigt':
                peak_params[f'p{i}_center'].set(value=peak_info['center'], min=peak_info['center']-50, max=peak_info['center']+50)
                peak_params[f'p{i}_amplitude'].set(value=peak_info['height']*peak_info['sigma']*np.sqrt(2*np.pi), min=0)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=0.001, max=100)
                peak_params[f'p{i}_gamma'].set(value=peak_info['sigma'], min=0.001, max=100)
                
            params.update(peak_params)
            
        return model, params
        
    def fit_spectrum(self, wavelengths, intensities, fit_params):
        """
        Fit a single spectrum
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
        fit_params : dict
            Fitting parameters
            
        Returns:
        --------
        ModelResult: Fitting result
        """
        # Create composite model
        model, params = self.create_composite_model(fit_params)
        
        if model is None:
            raise ValueError("Model creation failed")
        
        # Perform fitting
        result = model.fit(intensities, params, x=wavelengths)
        
        return result
        
    def fit_all_spectra(self, wavelengths, data_matrix, timestamps, fit_params, max_workers=None, use_smart_init=True, progress_callback=None):
        """
        Fit all spectra using parallel processing with smart parameter initialization
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        data_matrix : array
            Matrix of intensity values (time x wavelength)
        timestamps : array
            Time values
        fit_params : dict
            Fitting parameters
        max_workers : int, optional
            Maximum number of worker processes
        use_smart_init : bool
            Use previous fitting results to initialize nearby fits
            
        Returns:
        --------
        dict: Dictionary of fitting results
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(timestamps))
            
        # Store for smart initialization
        self.previous_results = {}
        
        # Prepare arguments for parallel processing
        fit_args = []
        for i, (time, spectrum) in enumerate(zip(timestamps, data_matrix)):
            # For smart initialization, include nearby results
            smart_params = fit_params.copy() if not use_smart_init else self._get_smart_init_params(fit_params, i, timestamps)
            fit_args.append((i, time, wavelengths, spectrum, smart_params, use_smart_init))
            
        # Perform parallel fitting with progress tracking
        results = {}
        completed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fitting tasks
            future_to_idx = {
                executor.submit(self._fit_single_spectrum_worker_smart, args): args[0] 
                for args in fit_args
            }
            
            # Collect results with progress bar and smart updates
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    # Store successful results for smart initialization
                    if result and result.get('success', False) and use_smart_init:
                        self.previous_results[idx] = result
                        
                    completed_count += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(completed_count, len(fit_args))
                        
                except Exception as e:
                    print(f"Error fitting spectrum {idx}: {e}")
                    results[idx] = None
                    completed_count += 1
                    
                    if progress_callback:
                        progress_callback(completed_count, len(fit_args))
                    
        return results
        
    def _get_smart_init_params(self, base_params, current_idx, timestamps, search_radius=5):
        """
        Get smart initialization parameters based on nearby successful fits
        
        Parameters:
        -----------
        base_params : dict
            Base fitting parameters
        current_idx : int
            Current time index
        timestamps : array
            Time values
        search_radius : int
            Number of nearby indices to search for good initial values
            
        Returns:
        --------
        dict: Optimized initial parameters
        """
        if not hasattr(self, 'previous_results') or not self.previous_results:
            return base_params
            
        # Find closest successful fit
        closest_result = None
        min_distance = float('inf')
        
        for idx, result in self.previous_results.items():
            if result and result.get('success', False):
                distance = abs(idx - current_idx)
                if distance < min_distance and distance <= search_radius:
                    min_distance = distance
                    closest_result = result
                    
        if closest_result is None:
            return base_params
            
        # Extract parameters from closest result
        smart_params = base_params.copy()
        closest_fitted_params = closest_result.get('parameters', {})
        
        # Update peak model parameters with fitted values
        for i, peak_model in enumerate(smart_params['peak_models']):
            peak_prefix = f'p{i}_'
            
            # Update center
            center_param = f'{peak_prefix}center'
            if center_param in closest_fitted_params:
                peak_model['center'] = closest_fitted_params[center_param]['value']
                
            # Update sigma
            sigma_param = f'{peak_prefix}sigma'
            if sigma_param in closest_fitted_params:
                peak_model['sigma'] = closest_fitted_params[sigma_param]['value']
                
            # Update height (converted from amplitude)
            amplitude_param = f'{peak_prefix}amplitude'
            if amplitude_param in closest_fitted_params and sigma_param in closest_fitted_params:
                amplitude = closest_fitted_params[amplitude_param]['value']
                sigma = closest_fitted_params[sigma_param]['value']
                if sigma > 0:
                    height = amplitude / (sigma * np.sqrt(2 * np.pi))
                    peak_model['height'] = height
                    
        return smart_params
        
    @staticmethod
    def _fit_single_spectrum_worker_smart(args):
        """
        Enhanced worker function for parallel fitting with smart initialization
        
        Parameters:
        -----------
        args : tuple
            (index, time, wavelengths, intensities, fit_params, use_smart_init)
            
        Returns:
        --------
        dict: Fitting result summary
        """
        idx, time, wavelengths, intensities, fit_params, use_smart_init = args
        
        try:
            # Create fitting instance
            fitter = FittingModels()
            
            # Perform fitting
            result = fitter.fit_spectrum(wavelengths, intensities, fit_params)
            
            # Extract key results with additional calculated parameters
            fit_summary = {
                'index': idx,
                'time': time,
                'success': result.success,
                'r_squared': getattr(result, 'rsquared', None),
                'chi_squared': getattr(result, 'chisqr', None),
                'reduced_chi_squared': result.redchi if hasattr(result, 'redchi') else None,
                'aic': getattr(result, 'aic', None),
                'bic': getattr(result, 'bic', None),
                'parameters': {},
                'fitted_curve': result.best_fit,
                'residuals': result.residual,
                'smart_init_used': use_smart_init
            }
            
            # Extract parameter values
            for param_name, param in result.params.items():
                fit_summary['parameters'][param_name] = {
                    'value': param.value,
                    'stderr': param.stderr,
                    'min': param.min,
                    'max': param.max
                }
            
            return fit_summary
            
        except Exception as e:
            return {
                'index': idx,
                'time': time,
                'success': False,
                'error': str(e),
                'parameters': {},
                'fitted_curve': None,
                'residuals': None,
                'smart_init_used': use_smart_init
            }
            
    def extract_peak_parameters(self, fit_results):
        """
        Extract peak parameters from fitting results
        
        Parameters:
        -----------
        fit_results : dict
            Dictionary of fitting results
            
        Returns:
        --------
        pandas.DataFrame: Peak parameters over time
        """
        import pandas as pd
        
        # Initialize lists to store parameters
        data_rows = []
        
        for idx, result in fit_results.items():
            if result is None or not result.get('success', False):
                continue
                
            row = {
                'index': result['index'],
                'time': result['time'],
                'r_squared': result.get('r_squared', np.nan),
                'chi_squared': result.get('chi_squared', np.nan),
                'aic': result.get('aic', np.nan),
                'bic': result.get('bic', np.nan)
            }
            
            # Extract peak parameters
            for param_name, param_data in result['parameters'].items():
                if any(x in param_name for x in ['center', 'amplitude', 'sigma', 'gamma', 'height']):
                    row[param_name] = param_data['value']
                    row[f"{param_name}_stderr"] = param_data.get('stderr', np.nan)
                    
            data_rows.append(row)
            
        return pd.DataFrame(data_rows)
        
    def create_model_summary(self, fit_params):
        """
        Create a summary of the fitting model
        
        Parameters:
        -----------
        fit_params : dict
            Fitting parameters
            
        Returns:
        --------
        dict: Model summary
        """
        summary = {
            'background_model': fit_params['background_model'],
            'peak_models': [],
            'total_parameters': 0
        }
        
        # Background parameters
        if fit_params['background_model'] != 'None':
            if fit_params['background_model'] == 'Linear':
                summary['total_parameters'] += 2  # slope + intercept
            elif fit_params['background_model'] == 'Polynomial':
                summary['total_parameters'] += fit_params['poly_degree'] + 1
            elif fit_params['background_model'] == 'Exponential':
                summary['total_parameters'] += 3  # amplitude, decay, offset
                
        # Peak parameters
        for i, peak_info in enumerate(fit_params['peak_models']):
            peak_summary = {
                'index': i,
                'type': peak_info['type'],
                'initial_center': peak_info['center'],
                'initial_height': peak_info['height'],
                'initial_sigma': peak_info['sigma']
            }
            
            if peak_info['type'] in ['Gaussian', 'Lorentzian']:
                summary['total_parameters'] += 3  # center, amplitude, sigma
            elif peak_info['type'] == 'Voigt':
                summary['total_parameters'] += 4  # center, amplitude, sigma, gamma
                
            summary['peak_models'].append(peak_summary)
            
        return summary


# =============================================================================
# FITTING ENGINE
# =============================================================================

class FittingEngine:
    """Orchestrates fitting operations"""
    
    def __init__(self):
        self.fitting_models = FittingModels()
        self.peak_detection = PeakDetector()
        
        # Fitting state
        self.fit_params = None
        self.fitting_results = {}
        self.current_fit_result = None
        
    def detect_peaks(self, wavelengths, intensities, **detection_params):
        """
        Detect peaks in spectrum
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
        **detection_params : dict
            Peak detection parameters
            
        Returns:
        --------
        list: Detected peak information
        """
        debug_print(f"Detecting peaks with params: {detection_params}", "FITTING")
        
        peaks = self.peak_detection.detect_peaks(
            wavelengths, 
            intensities, 
            **detection_params
        )
        
        debug_print(f"Detected {len(peaks)} peaks", "FITTING")
        return peaks
    
    def create_fit_parameters(self, peak_models, background_model='Linear', poly_degree=2):
        """
        Create fitting parameters dictionary
        
        Parameters:
        -----------
        peak_models : list
            List of peak model dictionaries
        background_model : str
            Background model type
        poly_degree : int
            Polynomial degree for polynomial background
            
        Returns:
        --------
        dict: Fitting parameters
        """
        self.fit_params = {
            'peak_models': peak_models,
            'background_model': background_model,
            'poly_degree': poly_degree
        }
        
        debug_print(f"Created fit parameters: {len(peak_models)} peaks, bg={background_model}", "FITTING")
        return self.fit_params
    
    def fit_current_spectrum(self, wavelengths, intensities):
        """
        Fit current spectrum
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
            
        Returns:
        --------
        ModelResult: Fitting result
        """
        if self.fit_params is None:
            raise ValueError("Fit parameters not set")
        
        debug_print("Fitting current spectrum", "FITTING")
        
        result = self.fitting_models.fit_spectrum(
            wavelengths,
            intensities,
            self.fit_params
        )
        
        self.current_fit_result = result
        
        debug_print(f"Fit complete: R²={result.rsquared:.4f}, success={result.success}", "FITTING")
        return result
    
    def fit_all_spectra(self, wavelengths, data_matrix, timestamps, 
                   fit_range=None, max_workers=None, use_smart_init=None, progress_callback=None):
        """
        Fit all spectra in batch
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        data_matrix : array
            Data matrix (time x wavelength)
        timestamps : array
            Time values
        fit_range : tuple, optional
            (start_idx, end_idx) for partial fitting
        max_workers : int, optional
            Number of parallel workers
        use_smart_init : bool, optional
            Use smart initialization
            
        Returns:
        --------
        dict: Fitting results
        """
        if self.fit_params is None:
            raise ValueError("Fit parameters not set")
        
        # Handle range selection
        if fit_range is not None:
            start_idx, end_idx = fit_range
            data_subset = data_matrix[start_idx:end_idx+1, :]
            time_subset = timestamps[start_idx:end_idx+1]
            debug_print(f"Batch fitting range: {start_idx} to {end_idx}", "FITTING")
        else:
            data_subset = data_matrix
            time_subset = timestamps
            debug_print("Batch fitting all spectra", "FITTING")
        
        # Use config defaults if not specified
        if max_workers is None:
            max_workers = config.MAX_WORKERS if config.USE_PARALLEL_FITTING else 1
        
        if use_smart_init is None:
            use_smart_init = config.USE_SMART_INIT
        
        debug_print(f"Starting batch fit: {len(time_subset)} spectra, workers={max_workers}, smart_init={use_smart_init}", "FITTING")
        
        results = self.fitting_models.fit_all_spectra(
            wavelengths,
            data_subset,
            time_subset,
            self.fit_params,
            max_workers=max_workers,
            use_smart_init=use_smart_init,
            progress_callback=progress_callback
        )
        
        # Store results (offset indices if fitting a range)
        if fit_range is not None:
            start_idx = fit_range[0]
            offset_results = {}
            for idx, result in results.items():
                offset_results[idx + start_idx] = result
            self.fitting_results.update(offset_results)
        else:
            self.fitting_results = results
        
        successful_fits = sum(1 for r in results.values() if r and r.get('success', False))
        debug_print(f"Batch fitting complete: {successful_fits}/{len(results)} successful", "FITTING")
        
        return self.fitting_results
    
    def get_fitting_result(self, time_idx):
        """
        Get fitting result for specific time index
        
        Parameters:
        -----------
        time_idx : int
            Time index
            
        Returns:
        --------
        dict or None: Fitting result
        """
        return self.fitting_results.get(time_idx, None)
    
    def has_fitting_results(self):
        """Check if any fitting results are available"""
        return len(self.fitting_results) > 0
    
    def clear_fitting_results(self):
        """Clear all fitting results"""
        debug_print("Clearing fitting results", "FITTING")
        self.fitting_results = {}
        self.current_fit_result = None
    
    def extract_peak_parameters(self):
        """
        Extract peak parameters from all fitting results
        
        Returns:
        --------
        pandas.DataFrame: Peak parameters
        """
        if not self.has_fitting_results():
            return None
        
        debug_print("Extracting peak parameters from results", "FITTING")
        return self.fitting_models.extract_peak_parameters(self.fitting_results)
    
    def get_model_summary(self):
        """
        Get summary of current fitting model
        
        Returns:
        --------
        dict: Model summary
        """
        if self.fit_params is None:
            return None
        
        return self.fitting_models.create_model_summary(self.fit_params)
    
    def update_peak_parameter(self, peak_idx, param_name, value):
        """
        Update a specific peak parameter
        
        Parameters:
        -----------
        peak_idx : int
            Peak index
        param_name : str
            Parameter name ('center', 'height', 'sigma')
        value : float
            New value
        """
        if self.fit_params is None or peak_idx >= len(self.fit_params['peak_models']):
            raise ValueError("Invalid peak index or no fit parameters")
        
        self.fit_params['peak_models'][peak_idx][param_name] = value
        debug_print(f"Updated peak {peak_idx} {param_name} to {value}", "FITTING")