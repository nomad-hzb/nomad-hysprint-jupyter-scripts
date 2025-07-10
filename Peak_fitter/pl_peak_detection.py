# pl_peak_detection.py
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.optimize import curve_fit

class PLPeakDetection:
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
        
    def detect_peaks_with_validation(self, wavelengths, intensities, validation_method='snr'):
        """
        Detect peaks with validation to reduce false positives
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
        validation_method : str
            Method for peak validation ('snr', 'shape', 'both')
            
        Returns:
        --------
        list: Validated peak information
        """
        # Initial peak detection
        peaks = self.detect_peaks_advanced(wavelengths, intensities)
        
        # Apply validation
        validated_peaks = []
        
        for peak in peaks:
            is_valid = True
            
            # Signal-to-noise ratio validation
            if validation_method in ['snr', 'both']:
                snr = self._calculate_peak_snr(intensities, peak['index'])
                if snr < 3:  # Minimum SNR threshold
                    is_valid = False
                    
            # Peak shape validation
            if validation_method in ['shape', 'both']:
                shape_quality = self._validate_peak_shape(intensities, peak['index'])
                if shape_quality < 0.7:  # Minimum shape quality
                    is_valid = False
                    
            if is_valid:
                validated_peaks.append(peak)
                
        return validated_peaks
        
    def _calculate_peak_snr(self, intensities, peak_idx):
        """Calculate signal-to-noise ratio for a peak"""
        # Signal is the peak height
        signal = intensities[peak_idx]
        
        # Noise is estimated from surrounding baseline
        search_range = 20
        left_start = max(0, peak_idx - search_range - 10)
        left_end = max(0, peak_idx - search_range)
        right_start = min(len(intensities), peak_idx + search_range)
        right_end = min(len(intensities), peak_idx + search_range + 10)
        
        baseline_values = np.concatenate([
            intensities[left_start:left_end],
            intensities[right_start:right_end]
        ])
        
        if len(baseline_values) > 0:
            noise = np.std(baseline_values)
            baseline = np.mean(baseline_values)
            return (signal - baseline) / noise if noise > 0 else float('inf')
        else:
            return float('inf')
            
    def _validate_peak_shape(self, intensities, peak_idx):
        """Validate peak shape quality"""
        # Extract local region around peak
        search_range = 10
        start_idx = max(0, peak_idx - search_range)
        end_idx = min(len(intensities), peak_idx + search_range + 1)
        
        local_intensities = intensities[start_idx:end_idx]
        local_peak_idx = peak_idx - start_idx
        
        # Check if it's actually a maximum
        if local_peak_idx <= 0 or local_peak_idx >= len(local_intensities) - 1:
            return 0.0
            
        # Calculate symmetry
        left_values = local_intensities[:local_peak_idx]
        right_values = local_intensities[local_peak_idx+1:]
        
        # Pad to same length
        min_len = min(len(left_values), len(right_values))
        if min_len < 2:
            return 0.0
            
        left_truncated = left_values[-min_len:]
        right_truncated = right_values[:min_len]
        
        # Calculate correlation between left and right sides (flipped)
        correlation = np.corrcoef(left_truncated, right_truncated[::-1])[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0