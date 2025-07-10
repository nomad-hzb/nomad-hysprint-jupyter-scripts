# pl_fitting_models.py
import numpy as np
from lmfit import Model, Parameters, CompositeModel
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel, LinearModel, PolynomialModel, ExponentialModel
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

class PLFittingModels:
    """Class to handle fitting of photoluminescence spectra"""
    
    def __init__(self):
        self.available_peak_models = {
            'Gaussian': GaussianModel,
            'Voigt': VoigtModel,
            'Lorentzian': LorentzianModel
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
        for i, peak_info in enumerate(fit_params['peak_models']):
            peak_model_class = self.available_peak_models[peak_info['type']]
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
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=1, max=100)
            elif peak_info['type'] == 'Lorentzian':
                peak_params[f'p{i}_center'].set(value=peak_info['center'], min=peak_info['center']-50, max=peak_info['center']+50)
                peak_params[f'p{i}_amplitude'].set(value=peak_info['height']*peak_info['sigma']*np.pi, min=0)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=1, max=100)
            elif peak_info['type'] == 'Voigt':
                peak_params[f'p{i}_center'].set(value=peak_info['center'], min=peak_info['center']-50, max=peak_info['center']+50)
                peak_params[f'p{i}_amplitude'].set(value=peak_info['height']*peak_info['sigma']*np.sqrt(2*np.pi), min=0)
                peak_params[f'p{i}_sigma'].set(value=peak_info['sigma'], min=1, max=100)
                peak_params[f'p{i}_gamma'].set(value=peak_info['sigma'], min=0.1, max=100)
                
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
        
        # Perform fitting
        result = model.fit(intensities, params, x=wavelengths)
        
        return result
        
    def fit_all_spectra(self, wavelengths, data_matrix, timestamps, fit_params, max_workers=None, use_smart_init=True):
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
            for future in tqdm(as_completed(future_to_idx), total=len(fit_args), desc="Fitting spectra"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    # Store successful results for smart initialization
                    if result and result.get('success', False) and use_smart_init:
                        self.previous_results[idx] = result
                        
                    completed_count += 1
                    
                    # Print progress updates
                    if completed_count % max(1, len(fit_args) // 10) == 0:
                        success_rate = sum(1 for r in results.values() if r and r.get('success', False)) / completed_count * 100
                        print(f"Progress: {completed_count}/{len(fit_args)} ({success_rate:.1f}% success rate)")
                        
                except Exception as e:
                    print(f"Error fitting spectrum {idx}: {e}")
                    results[idx] = None
                    completed_count += 1
                    
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
            fitter = PLFittingModels()
            
            # Perform fitting
            result = fitter.fit_spectrum(wavelengths, intensities, fit_params)
            
            # Extract key results with additional calculated parameters
            fit_summary = {
                'index': idx,
                'time': time,
                'success': result.success,
                'r_squared': result.rsquared if hasattr(result, 'rsquared') else None,
                'chi_squared': result.chisqr,
                'reduced_chi_squared': result.redchi if hasattr(result, 'redchi') else None,
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
                
            # Calculate additional peak properties
            fit_summary['peak_properties'] = fitter._calculate_peak_properties(result.params)
                
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
            
    def _calculate_peak_properties(self, fitted_params):
        """Calculate additional peak properties from fitted parameters"""
        properties = {}
        
        # Group parameters by peak
        peak_data = {}
        for param_name, param in fitted_params.items():
            if param_name.startswith('p') and '_' in param_name:
                peak_id = param_name.split('_')[0]
                param_type = '_'.join(param_name.split('_')[1:])
                
                if peak_id not in peak_data:
                    peak_data[peak_id] = {}
                peak_data[peak_id][param_type] = param.value
                
        # Calculate properties for each peak
        for peak_id, params in peak_data.items():
            if 'center' in params and 'amplitude' in params and 'sigma' in params:
                center = params['center']
                amplitude = params['amplitude']
                sigma = params['sigma']
                
                # Calculate height
                height = amplitude / (sigma * np.sqrt(2 * np.pi))
                
                # Calculate FWHM
                fwhm = 2.355 * sigma
                
                # Area under curve is the amplitude for Gaussian
                area = amplitude
                
                properties[peak_id] = {
                    'center': center,
                    'height': height,
                    'amplitude': amplitude,
                    'sigma': sigma,
                    'fwhm': fwhm,
                    'area_under_curve': area,
                    'width_at_half_max': fwhm,
                    'width_at_10_percent': 4.29 * sigma
                }
                
        return properties
    
    @staticmethod
    def _fit_single_spectrum_worker(args):
        """
        Worker function for parallel fitting
        
        Parameters:
        -----------
        args : tuple
            (index, time, wavelengths, intensities, fit_params)
            
        Returns:
        --------
        dict: Fitting result summary
        """
        idx, time, wavelengths, intensities, fit_params = args
        
        try:
            # Create fitting instance
            fitter = PLFittingModels()
            
            # Perform fitting
            result = fitter.fit_spectrum(wavelengths, intensities, fit_params)
            
            # Extract key results
            fit_summary = {
                'index': idx,
                'time': time,
                'success': result.success,
                'r_squared': result.rsquared if hasattr(result, 'rsquared') else None,
                'chi_squared': result.chisqr,
                'aic': result.aic,
                'bic': result.bic,
                'parameters': {},
                'fitted_curve': result.best_fit,
                'residuals': result.residual
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
                'residuals': None
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
        
    def calculate_peak_properties(self, fit_results, wavelengths):
        """
        Calculate additional peak properties from fitting results
        
        Parameters:
        -----------
        fit_results : dict
            Dictionary of fitting results
        wavelengths : array
            Wavelength values
            
        Returns:
        --------
        pandas.DataFrame: Peak properties
        """
        import pandas as pd
        
        properties = []
        
        for idx, result in fit_results.items():
            if result is None or not result.get('success', False):
                continue
                
            prop = {
                'index': result['index'],
                'time': result['time']
            }
            
            # Calculate integrated intensity for each peak
            params = result['parameters']
            peak_count = len([p for p in params.keys() if 'center' in p and p.startswith('p')])
            
            for i in range(peak_count):
                if f'p{i}_amplitude' in params:
                    prop[f'peak_{i}_integrated_intensity'] = params[f'p{i}_amplitude']['value']
                    
                if f'p{i}_center' in params:
                    prop[f'peak_{i}_center'] = params[f'p{i}_center']['value']
                    
                if f'p{i}_sigma' in params:
                    prop[f'peak_{i}_fwhm'] = 2.355 * params[f'p{i}_sigma']['value']  # FWHM for Gaussian
                    
            properties.append(prop)
            
        return pd.DataFrame(properties)
        
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