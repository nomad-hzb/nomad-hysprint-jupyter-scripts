# fitting_engine.py
"""
Consolidated fitting module
Contains peak detection, model definitions, and fitting orchestration
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.optimize import curve_fit
from lmfit import Model, Parameters, CompositeModel
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel, LinearModel, PolynomialModel, ExponentialModel, SkewedGaussianModel, SkewedVoigtModel
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
            'Polynomial': PolynomialModel,
            'Skewed Gaussian': SkewedGaussianModel,
            'Skewed Voigt': SkewedVoigtModel
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
        center_bound = fit_params.get('center_bound', config.DEFAULT_CENTER_BOUND)

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

            def _amp_kwargs(h, s, factor):
                """Convert user height bounds to lmfit amplitude bounds."""
                kw = {'value': h * s * factor, 'min': peak_info.get('height_min', 0) * s * factor}
                if 'height_max' in peak_info:
                    kw['max'] = peak_info['height_max'] * s * factor
                return kw

            def _bounds(key_min, key_max, default_min, default_max, floor=1e-9):
                """Resolve (min, max) honoring user-provided bounds. If only one
                side is provided and the auto-default for the other side would
                invalidate it (e.g. auto min > user max), relax the auto side
                down to `floor` / up to +inf so the user's bound is preserved."""
                umin = peak_info.get(key_min)
                umax = peak_info.get(key_max)
                lo = umin if umin is not None else default_min
                hi = umax if umax is not None else default_max
                if umax is not None and umin is None and lo >= hi:
                    lo = floor
                if umin is not None and umax is None and hi <= lo:
                    hi = float('inf')
                return lo, hi

            if peak_info['type'] == 'Gaussian':
                c = peak_info['center']
                c_lo, c_hi = _bounds('center_min', 'center_max',
                                     max(c - center_bound, 1e-6), c + center_bound, floor=1e-6)
                peak_params[f'p{i}_center'].set(value=c, min=c_lo, max=c_hi)
                peak_params[f'p{i}_amplitude'].set(
                    **_amp_kwargs(peak_info['height'], peak_info['sigma'], np.sqrt(2 * np.pi)))
                s_lo, s_hi = _bounds('sigma_min', 'sigma_max', 0.00001, 100, floor=1e-9)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=s_lo, max=s_hi)
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
                c = peak_info['center']
                c_lo, c_hi = _bounds('center_min', 'center_max',
                                     max(c - center_bound, 1e-6), c + center_bound, floor=1e-6)
                peak_params[f'p{i}_center'].set(value=c, min=c_lo, max=c_hi)
                peak_params[f'p{i}_amplitude'].set(
                    **_amp_kwargs(peak_info['height'], peak_info['sigma'], np.pi))
                s_lo, s_hi = _bounds('sigma_min', 'sigma_max', 0.00001, 100, floor=1e-9)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=s_lo, max=s_hi)
            elif peak_info['type'] == 'Voigt':
                c = peak_info['center']
                c_lo, c_hi = _bounds('center_min', 'center_max',
                                     max(c - center_bound, 1e-6), c + center_bound, floor=1e-6)
                peak_params[f'p{i}_center'].set(value=c, min=c_lo, max=c_hi)
                peak_params[f'p{i}_amplitude'].set(
                    **_amp_kwargs(peak_info['height'], peak_info['sigma'], np.sqrt(2 * np.pi)))
                s_lo, s_hi = _bounds('sigma_min', 'sigma_max', 0.001, 100, floor=1e-9)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=s_lo, max=s_hi)
                g_lo, g_hi = _bounds('gamma_min', 'gamma_max', 0.001, 100, floor=1e-9)
                peak_params[f'p{i}_gamma'].set(value=peak_info['gamma'], min=g_lo, max=g_hi)
            elif peak_info['type'] == 'Skewed Gaussian':
                c = peak_info['center']
                c_lo, c_hi = _bounds('center_min', 'center_max',
                                     max(c - center_bound, 1e-6), c + center_bound, floor=1e-6)
                peak_params[f'p{i}_center'].set(value=c, min=c_lo, max=c_hi)
                peak_params[f'p{i}_amplitude'].set(
                    **_amp_kwargs(peak_info['height'], peak_info['sigma'], np.sqrt(2 * np.pi)))
                s_lo, s_hi = _bounds('sigma_min', 'sigma_max', 0.001, 100, floor=1e-9)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=s_lo, max=s_hi)
                g_lo, g_hi = _bounds('gamma_min', 'gamma_max', -10, 10, floor=-float('inf'))
                peak_params[f'p{i}_gamma'].set(value=peak_info.get('gamma', 0.0), min=g_lo, max=g_hi)
            elif peak_info['type'] == 'Skewed Voigt':
                c = peak_info['center']
                c_lo, c_hi = _bounds('center_min', 'center_max',
                                     max(c - center_bound, 1e-6), c + center_bound, floor=1e-6)
                peak_params[f'p{i}_center'].set(value=c, min=c_lo, max=c_hi)
                peak_params[f'p{i}_amplitude'].set(
                    **_amp_kwargs(peak_info['height'], peak_info['sigma'], np.sqrt(2 * np.pi)))
                s_lo, s_hi = _bounds('sigma_min', 'sigma_max', 0.001, 100, floor=1e-9)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=s_lo, max=s_hi)
                g_lo, g_hi = _bounds('gamma_min', 'gamma_max', -10, 10, floor=-float('inf'))
                peak_params[f'p{i}_gamma'].set(value=peak_info.get('gamma', 0.0), min=g_lo, max=g_hi)
                peak_params[f'p{i}_skew'].set(value=peak_info.get('skew', 0.0), min=-10, max=10)

            # Log resolved bounds per parameter, marking USER vs AUTO defaults
            _bound_keymap = {
                'center':    ('center_min', 'center_max'),
                'sigma':     ('sigma_min', 'sigma_max'),
                'gamma':     ('gamma_min', 'gamma_max'),
                'amplitude': ('height_min', 'height_max'),
            }
            _log_parts = []
            for _pname, (_kmin, _kmax) in _bound_keymap.items():
                _pkey = f'p{i}_{_pname}'
                if _pkey not in peak_params:
                    continue
                _p = peak_params[_pkey]
                _umin = '(USER)' if peak_info.get(_kmin) is not None else ''
                _umax = '(USER)' if peak_info.get(_kmax) is not None else ''
                _log_parts.append(
                    f"{_pname}={_p.value:.4g} ∈ [{_p.min:.4g}{_umin}, {_p.max:.4g}{_umax}]"
                )
            if _log_parts:
                debug_print(
                    f"p{i} ({peak_info['type']}) " + "  ".join(_log_parts),
                    "FITTING"
                )

            params.update(peak_params)

            if peak_info.get('fix_center') and f'p{i}_center' in params:
                params[f'p{i}_center'].set(vary=False)
            if peak_info.get('fix_height') and f'p{i}_amplitude' in params:
                params[f'p{i}_amplitude'].set(vary=False)
            if peak_info.get('fix_sigma') and f'p{i}_sigma' in params:
                params[f'p{i}_sigma'].set(vary=False)
            if peak_info.get('fix_gamma') and f'p{i}_gamma' in params:
                params[f'p{i}_gamma'].set(vary=False)

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

        # Strip non-finite values (nan/inf) before fitting
        finite_mask = np.isfinite(intensities)
        wl_fit = wavelengths[finite_mask]
        int_fit = intensities[finite_mask]

        if len(int_fit) == 0:
            raise ValueError("No finite data points to fit (all nan/inf)")

        # Perform fitting on finite points only
        result = model.fit(int_fit, params, x=wl_fit)

        # Attach the filtered x-axis so callers can reconstruct full-length arrays
        result.fit_x = wl_fit

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

        results = {}
        completed_count = 0

        if use_smart_init:
            # Sequential fitting: each frame uses the previous successful result as
            # start values. Parallelism is incompatible with this sequential dependency.
            last_successful_result = None
            for i, (time, spectrum) in enumerate(zip(timestamps, data_matrix)):
                current_params = self._apply_smart_init(fit_params, last_successful_result)
                args = (i, time, wavelengths, spectrum, current_params, True)
                result = self._fit_single_spectrum_worker_smart(args)
                results[i] = result
                if result and result.get('success', False):
                    last_successful_result = result
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, len(timestamps))
        else:
            # Parallel fitting when smart init is not needed
            fit_args = [
                (i, time, wavelengths, spectrum, fit_params, False)
                for i, (time, spectrum) in enumerate(zip(timestamps, data_matrix))
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._fit_single_spectrum_worker_smart, args): args[0]
                    for args in fit_args
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Error fitting spectrum {idx}: {e}")
                        results[idx] = None
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, len(fit_args))

        return results
        
    def _apply_smart_init(self, base_params, previous_result):
        """
        Return a copy of base_params with peak center/sigma/height seeded from
        the previous successful fit result. If previous_result is None the
        base_params are returned unchanged.
        """
        if previous_result is None:
            debug_print("Smart init: no previous result, using base params", "FITTING")
            return base_params

        import copy
        smart_params = copy.deepcopy(base_params)
        fitted_params = previous_result.get('parameters', {})

        for i, peak_model in enumerate(smart_params['peak_models']):
            prefix = f'p{i}_'

            center_key = f'{prefix}center'
            if center_key in fitted_params:
                old_center = peak_model['center']
                peak_model['center'] = fitted_params[center_key]['value']
                debug_print(f"Smart init p{i}: center {old_center:.2f} -> {peak_model['center']:.2f}", "FITTING")

            sigma_key = f'{prefix}sigma'
            if sigma_key in fitted_params:
                peak_model['sigma'] = fitted_params[sigma_key]['value']

            amplitude_key = f'{prefix}amplitude'
            if amplitude_key in fitted_params and sigma_key in fitted_params:
                amplitude = fitted_params[amplitude_key]['value']
                sigma = fitted_params[sigma_key]['value']
                if sigma > 0:
                    peak_model['height'] = amplitude / (sigma * np.sqrt(2 * np.pi))

        debug_print(f"Smart init: seeded from idx={previous_result.get('index')} t={previous_result.get('time', 0):.3f}", "FITTING")
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
                'peak_models': fit_params.get('peak_models', []),
                'fit_x': result.fit_x,
                'fitted_curve': result.best_fit,
                'residuals': result.residual,
                'components': result.eval_components() if hasattr(result, 'eval_components') else {},
                'smart_init_used': use_smart_init
            }

            # debug_print("Eval components: " + ", ".join(fit_summary['components']), "FITTING")
            
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
        self.fit_wavelengths = None  # Wavelengths used for last batch fit
        
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
    
    def create_fit_parameters(self, peak_models, background_model='Linear', poly_degree=2, center_bound=None, **kwargs):
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
            'poly_degree': poly_degree,
            'center_bound': center_bound,
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
        
        self.fit_wavelengths = wavelengths

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
            self.fitting_results = offset_results  # Replace, don't merge!
        else:
            self.fitting_results = results
        
        successful_fits = sum(1 for r in results.values() if r and r.get('success', False))
        debug_print(f"Batch fitting complete: {successful_fits}/{len(results)} successful", "FITTING")

        failed = {idx: r for idx, r in results.items() if not (r and r.get('success', False))}
        if failed:
            debug_print(f"Failed fits ({len(failed)}):", "FITTING")
            for idx, r in sorted(failed.items()):
                if r is None:
                    debug_print(f"  idx={idx}: result is None (future raised exception)", "FITTING")
                elif 'error' in r:
                    debug_print(f"  idx={idx} t={r.get('time', '?'):.3f}: exception: {r['error']}", "FITTING")
                else:
                    r2 = r.get('r_squared', float('nan'))
                    redchi = r.get('reduced_chi_squared', float('nan'))
                    # Detect parameters that hit their bounds
                    at_bounds = [
                        f"{name}={'min' if abs(p['value'] - p['min']) < 1e-6 else 'max'}"
                        for name, p in r.get('parameters', {}).items()
                        if abs(p['value'] - p['min']) < 1e-6 or abs(p['value'] - p['max']) < 1e-6
                    ]
                    bound_str = f", at bounds: {at_bounds}" if at_bounds else ""
                    debug_print(f"  idx={idx} t={r.get('time', '?'):.3f}: lmfit success=False "
                                f"(R²={r2:.4f}, redchi={redchi:.4f}{bound_str})", "FITTING")

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