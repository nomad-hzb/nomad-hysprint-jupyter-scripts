# data_manager.py
"""
Consolidated data handling module
Contains all data loading, parsing, and management functionality
"""
import numpy as np
import pandas as pd
import io
import os
import sys
import config
from utils import debug_print

parent_dir = os.path.dirname(os.getcwd())
utils_dir = os.path.join(parent_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
import access_token

# Log notebook usage
access_token.log_notebook_usage()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_axes_from_extent(extent, data):
    """
    Calculate axes from extent information
    
    Parameters:
    -----------
    extent : list
        [xmin, xmax, ymin, ymax]
    data : array
        Data matrix
        
    Returns:
    --------
    tuple: (xaxes, yaxes)
    """
    debug_print("Calculating axes from extent", "H5")
    [xmin, xmax, ymin, ymax] = extent
    nrows, ncols = data.shape
    xaxes = np.linspace(xmin, xmax, ncols)
    yaxes = np.linspace(ymin, ymax, nrows)
    return xaxes, yaxes


def get_h5_path_from_ipython():
    """
    Retrieve h5_path from IPython stored variables (ISA Voila integration)
    
    Returns:
    --------
    tuple: (h5_path, success) where success is True if path was found
    """
    try:
        debug_print("Attempting to retrieve h5_path from IPython store", "H5")
        
        # Use IPython magic to retrieve stored variable
        from IPython import get_ipython
        ipython = get_ipython()
        
        if ipython is not None:
            # Execute the magic command
            ipython.magic('store -r h5_path')
            
            # Try to access the variable from user namespace
            if 'h5_path' in ipython.user_ns:
                h5_path = ipython.user_ns['h5_path']
                debug_print(f"Found h5_path: {h5_path}", "H5")
                return h5_path, True
        
        debug_print("h5_path not found in IPython store", "H5")
        return None, False
        
    except Exception as e:
        debug_print(f"Error retrieving h5_path: {e}", "H5")
        return None, False

def sanitize_float(value, default=0.0):
    """
    Replace inf/nan with a default value
    
    Parameters:
    -----------
    value : float
        Value to sanitize
    default : float
        Default value to use if inf/nan
        
    Returns:
    --------
    float: Sanitized value
    """
    if not np.isfinite(value):
        return default
    return float(value)

def sanitize_array(arr, replace_with=0.0):
    """
    Replace all inf/nan values in array
    
    Parameters:
    -----------
    arr : array
        Array to sanitize
    replace_with : float
        Value to replace inf/nan with
        
    Returns:
    --------
    array: Sanitized array
    """
    arr = np.where(np.isinf(arr), np.nan, arr)
    arr = np.where(np.isnan(arr), replace_with, arr)
    return arr


# =============================================================================
# CSV DATA LOADER
# =============================================================================

class CSVDataLoader:
    """Class to handle loading and parsing of data files"""

    def __init__(self):
        self.data_matrix = None
        self.wavelengths = None
        self.timestamps = None
        self.header_info = {}

    def _normalize_timestamps_to_zero(self, timestamps):
        """Normalize timestamps to start at zero"""
        if len(timestamps) == 0:
            return timestamps
        
        first_timestamp = timestamps[0]
        debug_print(f"Normalizing timestamps - Original range: {timestamps[0]:.3f} - {timestamps[-1]:.3f}", "DATA")
        
        normalized_timestamps = timestamps - first_timestamp
        debug_print(f"Normalized time range: {normalized_timestamps[0]:.3f} - {normalized_timestamps[-1]:.3f}s", "DATA")
        
        return normalized_timestamps

    def load_data(self, file_content):
        """Load photoluminescence data from file content"""
        debug_print("=="*25, "DATA")
        debug_print("Starting CSV data load", "DATA")
        debug_print(f"File content type: {type(file_content)}", "DATA")
        debug_print(f"File content length: {len(file_content)} bytes", "DATA")
        
        # Convert bytes/memoryview to string with encoding detection
        if isinstance(file_content, bytes):
            raw_bytes = file_content
        elif isinstance(file_content, memoryview):
            raw_bytes = file_content.tobytes()
        else:
            content_str = str(file_content)
            raw_bytes = None
    
        if raw_bytes is not None:
            debug_print(f"First 200 bytes (raw): {raw_bytes[:200]}", "DATA")
            
            # Check for BOM to detect encoding
            if raw_bytes.startswith(b'\xff\xfe'):
                # UTF-16 LE BOM
                content_str = raw_bytes.decode('utf-16-le')
                debug_print("Detected UTF-16 LE encoding (BOM: \\xff\\xfe)", "DATA")
            elif raw_bytes.startswith(b'\xfe\xff'):
                # UTF-16 BE BOM
                content_str = raw_bytes.decode('utf-16-be')
                debug_print("Detected UTF-16 BE encoding (BOM: \\xfe\\xff)", "DATA")
            elif raw_bytes.startswith(b'\xef\xbb\xbf'):
                # UTF-8 BOM
                content_str = raw_bytes.decode('utf-8-sig')
                debug_print("Detected UTF-8 encoding with BOM", "DATA")
            else:
                # Default to UTF-8
                try:
                    content_str = raw_bytes.decode('utf-8')
                    debug_print("Decoded as UTF-8", "DATA")
                except UnicodeDecodeError:
                    # Try UTF-16 without BOM as fallback
                    try:
                        content_str = raw_bytes.decode('utf-16')
                        debug_print("Decoded as UTF-16 (no BOM detected)", "DATA")
                    except:
                        raise ValueError("Could not decode file - unsupported encoding")
        
        debug_print(f"String length: {len(content_str)} characters", "DATA")
        debug_print(f"First 200 chars: {content_str[:200]}", "DATA")

        # Parse the file
        lines = content_str.strip().split('\n')

        # Find the header row that contains "Wavelength"
        header_row_idx = None
        for i, line in enumerate(lines[:50]):  # Check first 50 lines
            line_stripped = line.strip()
            if any(keyword in line_stripped for keyword in ['Wavelength', 'wavelength', 'WAVELENGTH']):
                header_row_idx = i
                break

        if header_row_idx is None:
            raise ValueError("Could not find 'Wavelength' header row in the file")

        # Extract metadata from lines before the header
        self.header_info = self._extract_metadata(lines[:header_row_idx])

        # Parse header row to get timestamps
        header_line = lines[header_row_idx]
        header_parts = header_line.split(',')

        # Filter out empty strings and convert to float
        timestamp_strings = header_parts[1:]
        valid_timestamps = [x.strip() for x in timestamp_strings if x.strip()]
        # Filter out empty strings and convert to float
        timestamp_strings = header_parts[1:]
        valid_timestamps = [x.strip() for x in timestamp_strings if x.strip()]
        
        debug_print(f"Found {len(valid_timestamps)} valid timestamp strings", "DATA")
        debug_print(f"First timestamp string: '{valid_timestamps[0]}'", "DATA")
        debug_print(f"Last timestamp string: '{valid_timestamps[-1]}'", "DATA")
        
        timestamps_array = np.array([float(x) for x in valid_timestamps])
        debug_print(f"Converted to array, shape: {timestamps_array.shape}", "DATA")
        debug_print(f"Timestamp range: {timestamps_array.min():.3f} to {timestamps_array.max():.3f}", "DATA")
        
        # NORMALIZE TIMESTAMPS TO START AT ZERO
        self.timestamps = self._normalize_timestamps_to_zero(timestamps_array)
        debug_print(f"After normalization: {self.timestamps.min():.3f} to {self.timestamps.max():.3f}", "DATA")

        # Parse data rows
        wavelengths_list = []
        intensity_matrix = []

        data_lines = lines[header_row_idx + 1:]

        for line in data_lines:
            if not line.strip():
                continue

            parts = line.split(',')
            if len(parts) < 2:
                continue

            try:
                # First column is wavelength
                wavelength = float(parts[0])
                wavelengths_list.append(wavelength)

                # Remaining columns are intensities
                intensity_strings = parts[1:len(self.timestamps) + 1]
                intensities = []

                for intensity_str in intensity_strings:
                    if intensity_str.strip():
                        intensities.append(float(intensity_str.strip()))
                    else:
                        intensities.append(0.0)

                # Ensure we have the right number of intensities
                while len(intensities) < len(self.timestamps):
                    intensities.append(0.0)

                intensity_matrix.append(intensities[:len(self.timestamps)])

            except (ValueError, IndexError):
                continue

        # Convert to numpy arrays
        if len(wavelengths_list) == 0:
            raise ValueError("No wavelength data found")
        if len(intensity_matrix) == 0:
            raise ValueError("No intensity data found")

        self.wavelengths = np.array(wavelengths_list)
        intensity_array = np.array(intensity_matrix)
        
        # Transpose to have time as first dimension (time x wavelength)
        self.data_matrix = intensity_array.T
        
        debug_print(f"Intensity array shape before transpose: {intensity_array.shape}", "DATA")
        debug_print(f"Final data_matrix shape: {self.data_matrix.shape}", "DATA")
        
        # **CRITICAL: Sanitize infinity and NaN values**
        num_inf = np.sum(np.isinf(self.data_matrix))
        num_nan = np.sum(np.isnan(self.data_matrix))
        
        if num_inf > 0 or num_nan > 0:
            debug_print(f"Found {num_inf} inf and {num_nan} NaN values - replacing with 0", "DATA")
            self.data_matrix = sanitize_array(self.data_matrix, replace_with=0.0)
        
        debug_print(f"Intensity range: {self.data_matrix.min():.2f} to {self.data_matrix.max():.2f}", "DATA")
        debug_print("="*50, "DATA")
        
        return self.data_matrix, self.wavelengths, self.timestamps

    def _extract_metadata(self, metadata_lines):
        """Extract metadata from key,value pairs"""
        metadata = {}

        for line in metadata_lines:
            if ',' in line:
                parts = line.split(',', 1)  # Split only on first comma
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    metadata[key] = value

        return metadata

    def get_header_info(self):
        """Return header information as dictionary"""
        return self.header_info

    def get_data_info(self):
        """Return summary of loaded data"""
        if self.data_matrix is None:
            return "No data loaded"

        info = {
            'shape': self.data_matrix.shape,
            'time_points': len(self.timestamps),
            'wavelengths': len(self.wavelengths),
            'time_range': (self.timestamps.min(), self.timestamps.max()),
            'wavelength_range': (self.wavelengths.min(), self.wavelengths.max()),
            'intensity_range': (self.data_matrix.min(), self.data_matrix.max())
        }

        return info

    def validate_data(self):
        """Validate loaded data for common issues"""
        issues = []

        if self.data_matrix is None:
            issues.append("No data loaded")
            return issues

        # Check for NaN values
        if np.isnan(self.data_matrix).any():
            issues.append(f"Found {np.isnan(self.data_matrix).sum()} NaN values in data")

        # Check for negative intensities
        if (self.data_matrix < 0).any():
            issues.append(f"Found {(self.data_matrix < 0).sum()} negative intensity values")

        # Check wavelength ordering
        if not np.all(np.diff(self.wavelengths) > 0):
            issues.append("Wavelengths are not in ascending order")

        # Check time ordering
        if not np.all(np.diff(self.timestamps) > 0):
            issues.append("Timestamps are not in ascending order")

        return issues


# =============================================================================
# H5 DATA LOADER
# =============================================================================

class H5DataLoader:
    """Handler for loading data from H5 files"""
    
    def __init__(self):
        self.h5_path = None
        self.data_available = False
        
    def check_for_h5_data(self):
        """
        Check if H5 data is available from ISA Voila
        
        Returns:
        --------
        bool: True if H5 data is available
        """
        self.h5_path, self.data_available = get_h5_path_from_ipython()
        return self.data_available
    
    def load_h5_data(self, mode, h5_path=None):
        """
        Load data from H5 file based on mode
        
        Parameters:
        -----------
        mode : str
            Data mode ('pl_raw', 'pl_binned', 'giwaxs')
        h5_path : str, optional
            Path to H5 file. If None, uses stored path
            
        Returns:
        --------
        tuple: (data_matrix, wavelengths, timestamps)
        """
        import h5py
        
        if h5_path is None:
            h5_path = self.h5_path
            
        if h5_path is None:
            raise ValueError("No H5 file path available")
        
        debug_print(f"Loading H5 data in mode: {mode}", "H5")
        
        with h5py.File(h5_path, "r") as f:
            if mode == "pl_raw":
                timestamps = f[config.H5_PATHS['pl_raw']['timestamps']][()]
                data_matrix = f[config.H5_PATHS['pl_raw']['data']][()]
                wavelengths = f[config.H5_PATHS['pl_raw']['wavelengths']][()]
                
            elif mode == "pl_binned":
                extent = f[config.H5_PATHS['pl_binned']['extent']][()]
                data_matrix = f[config.H5_PATHS['pl_binned']['data']][()].T
                timestamps, wavelengths = get_axes_from_extent(extent, data_matrix)
                
            elif mode == "giwaxs":
                timestamps = f[config.H5_PATHS['giwaxs']['timestamps']][()]
                data_matrix = f[config.H5_PATHS['giwaxs']['data']][()]
                wavelengths = f[config.H5_PATHS['giwaxs']['wavelengths']][()][0] / 10
                
            else:
                raise ValueError(f"Unknown H5 mode: {mode}")
        
        # Clean up NaN values in timestamps
        if np.isnan(timestamps[-1]):
            debug_print("Removing NaN from last timestamp entry", "H5")
            timestamps = timestamps[:-1]
        
        debug_print(f"Loaded H5 data: {data_matrix.shape}, {len(wavelengths)} wavelengths, {len(timestamps)} times", "H5")
        
        return data_matrix, wavelengths, timestamps


# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager:
    """Manages data loading, storage, and validation"""
    
    def __init__(self):
        self.csv_loader = CSVDataLoader()
        self.h5_loader = H5DataLoader()
        
        # Data storage
        self.data_matrix = None
        self.wavelengths = None
        self.timestamps = None
        self.current_time_idx = 0
        self.current_spectrum = None
        
        # Data source tracking
        self.data_source = None  # 'csv', 'h5', or None
        self.h5_mode = None
        
        # Check for H5 data availability
        self.h5_available = self.h5_loader.check_for_h5_data()
        
    def is_h5_available(self):
        """Check if H5 data source is available"""
        return self.h5_available
    
    def load_from_h5(self, mode):
        """
        Load data from H5 file
        
        Parameters:
        -----------
        mode : str
            H5 data mode
            
        Returns:
        --------
        bool: True if successful
        """
        try:
            debug_print(f"Loading data from H5 (mode: {mode})", "DATA")
            
            self.data_matrix, self.wavelengths, self.timestamps = \
                self.h5_loader.load_h5_data(mode)
            
            self.data_source = 'h5'
            self.h5_mode = mode
            self.current_time_idx = 0
            self.current_spectrum = self.data_matrix[0, :]
            
            debug_print(f"H5 data loaded successfully: {self.data_matrix.shape}", "DATA")
            return True
            
        except Exception as e:
            debug_print(f"Error loading H5 data: {e}", "DATA")
            raise
    
    def load_from_file(self, file_content):
        """
        Load data from uploaded file
        
        Parameters:
        -----------
        file_content : bytes
            File content from upload widget
            
        Returns:
        --------
        bool: True if successful
        """
        try:
            debug_print("Loading data from uploaded file", "DATA")
            
            self.data_matrix, self.wavelengths, self.timestamps = \
                self.csv_loader.load_data(file_content)
            
            self.data_source = 'csv'
            self.h5_mode = None
            self.current_time_idx = 0
            self.current_spectrum = self.data_matrix[0, :]
            
            debug_print(f"CSV data loaded successfully: {self.data_matrix.shape}", "DATA")
            return True
            
        except Exception as e:
            debug_print(f"Error loading CSV data: {e}", "DATA")
            raise
    
    def is_data_loaded(self):
        """Check if data is currently loaded"""
        return self.data_matrix is not None
    
    def get_data_info(self):
        """Get information about loaded data"""
        if not self.is_data_loaded():
            return {"status": "No data loaded"}
        
        # Use sanitize_float for min/max values
        data_min = sanitize_float(self.data_matrix.min(), default=0.0)
        data_max = sanitize_float(self.data_matrix.max(), default=1000.0)
        
        return {
            "status": "Data loaded",
            "source": self.data_source,
            "shape": self.data_matrix.shape,
            "time_points": len(self.timestamps),
            "wavelengths": len(self.wavelengths),
            "time_range": (sanitize_float(self.timestamps.min()), sanitize_float(self.timestamps.max())),
            "wavelength_range": (sanitize_float(self.wavelengths.min()), sanitize_float(self.wavelengths.max())),
            "intensity_range": (data_min, data_max)
        }
    
    def get_spectrum_at_time(self, time_idx):
        """
        Get spectrum at specific time index
        
        Parameters:
        -----------
        time_idx : int
            Time index
            
        Returns:
        --------
        array: Intensity spectrum
        """
        if not self.is_data_loaded():
            raise ValueError("No data loaded")
        
        if time_idx < 0 or time_idx >= len(self.timestamps):
            raise ValueError(f"Time index {time_idx} out of range")
        
        return self.data_matrix[time_idx, :]
    
    def set_current_time(self, time_idx):
        """
        Set current time index and update current spectrum
        
        Parameters:
        -----------
        time_idx : int
            Time index
        """
        if not self.is_data_loaded():
            raise ValueError("No data loaded")
        
        # Validate and clip index
        time_idx = max(0, min(time_idx, len(self.timestamps) - 1))
        
        self.current_time_idx = time_idx
        self.current_spectrum = self.data_matrix[time_idx, :]
        
        debug_print(f"Current time set to index {time_idx} (t={self.timestamps[time_idx]:.3f}s)", "DATA")
    
    def get_current_spectrum(self):
        """Get current spectrum"""
        return self.current_spectrum
    
    def get_current_time_value(self):
        """Get current time value in seconds"""
        if not self.is_data_loaded():
            return None
        return self.timestamps[self.current_time_idx]
    
    def get_time_range(self):
        """Get valid time index range"""
        if not self.is_data_loaded():
            return (0, 0)
        return (0, len(self.timestamps) - 1)
    
    def validate_data(self):
        """
        Validate loaded data for issues
        
        Returns:
        --------
        list: List of validation issues (empty if no issues)
        """
        if not self.is_data_loaded():
            return ["No data loaded"]
        
        return self.csv_loader.validate_data()
    
    def get_header_info(self):
        """Get header/metadata information"""
        if self.data_source == 'csv':
            return self.csv_loader.get_header_info()
        elif self.data_source == 'h5':
            return {
                "source": "H5 file",
                "mode": self.h5_mode,
                "h5_path": self.h5_loader.h5_path
            }
        return {}

    def set_wavelength_unit_label(self, unit):
        """
        Set the wavelength unit label without changing data
        This is for display purposes only when different data types are loaded
        
        Parameters:
        -----------
        unit : str
            'nm' or 'angstrom'
        
        Returns:
        --------
        bool: True if successful
        """
        if unit not in ['nm', 'angstrom']:
            debug_print(f"Invalid unit: {unit}", "DATA")
            return False
        
        debug_print(f"Setting wavelength unit label to {unit}", "DATA")
        return True

    def convert_wavelength_to_energy(self):
        """
        Convert wavelength (nm) to energy (eV)
        E (eV) = 1239.8 / λ (nm)
        
        Returns:
        --------
        bool: True if conversion successful
        """
        if not self.is_data_loaded():
            debug_print("Cannot convert: no data loaded", "DATA")
            return False
        
        debug_print(f"Converting wavelength to energy", "DATA")
        debug_print(f"Original range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} nm", "DATA")
        
        # E (eV) = 1239.8 / λ (nm)
        # Note: energy is inversely proportional, so order reverses
        self.wavelengths = 1239.8 / self.wavelengths
        # Reverse the array so it's still in ascending order
        self.wavelengths = self.wavelengths[::-1]
        # Also need to reverse data matrix wavelength dimension
        self.data_matrix = self.data_matrix[:, ::-1]
        
        debug_print(f"Converted range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} eV", "DATA")
        
        return True

    def convert_wavelengths_to_angstrom(self):
        """
        Convert wavelength axis from nm to Angstrom
        1 nm = 10 Angstrom
        
        Returns:
        --------
        bool: True if conversion successful
        """
        if not self.is_data_loaded():
            debug_print("Cannot convert: no data loaded", "DATA")
            return False
        
        debug_print(f"Converting wavelengths from nm to Angstrom", "DATA")
        debug_print(f"Original range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} nm", "DATA")
        
        # Convert nm to Angstrom (multiply by 10)
        self.wavelengths = self.wavelengths * 10
        
        debug_print(f"Converted range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} Å", "DATA")
        
        return True

    def convert_wavelengths_to_nm(self):
        """
        Convert wavelength axis from Angstrom back to nm
        1 nm = 10 Angstrom
        
        Returns:
        --------
        bool: True if conversion successful
        """
        if not self.is_data_loaded():
            debug_print("Cannot convert: no data loaded", "DATA")
            return False
        
        debug_print(f"Converting wavelengths from Angstrom to nm", "DATA")
        debug_print(f"Original range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} Å", "DATA")
        
        # Convert Angstrom to nm (divide by 10)
        self.wavelengths = self.wavelengths / 10
        
        debug_print(f"Converted range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} nm", "DATA")
        
        return True
        
    def convert_energy_to_wavelength(self):
        """
        Convert energy (eV) to wavelength (nm)
        λ (nm) = 1239.8 / E (eV)
        
        Returns:
        --------
        bool: True if conversion successful
        """
        if not self.is_data_loaded():
            debug_print("Cannot convert: no data loaded", "DATA")
            return False
        
        debug_print(f"Converting energy to wavelength", "DATA")
        debug_print(f"Original range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} eV", "DATA")
        
        # λ (nm) = 1239.8 / E (eV)
        self.wavelengths = 1239.8 / self.wavelengths
        # Reverse back to original order
        self.wavelengths = self.wavelengths[::-1]
        self.data_matrix = self.data_matrix[:, ::-1]
        
        debug_print(f"Converted range: {self.wavelengths.min():.2f} - {self.wavelengths.max():.2f} nm", "DATA")
        
        return True