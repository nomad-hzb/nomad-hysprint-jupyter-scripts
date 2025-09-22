# pl_data_loader.py
import numpy as np
import pandas as pd
import io


def get_axes_from_extent(extent, data):
    print("Start get axes from extent")
    [xmin, xmax, ymin, ymax] = extent
    nrwos, ncols = data.shape
    xaxes = np.linspace(xmin, xmax, ncols)
    yaxes = np.linspace(ymin, ymax, nrwos)
    return xaxes, yaxes


class PLDataLoader:
    """Class to handle loading and parsing of photoluminescence data files"""
    
    def __init__(self):
        self.data_matrix = None
        self.wavelengths = None
        self.timestamps = None
        self.header_info = {}
        
    def load_data(self, file_content):
        """
        Load photoluminescence data from file content
        
        Expected format:
        - Metadata lines (key,value pairs)
        - Row starting with "Wavelength (nm)" followed by time values
        - Subsequent rows: wavelength,intensity1,intensity2,...
        
        Parameters:
        -----------
        file_content : bytes or memoryview
            Raw file content from upload
            
        Returns:
        --------
        tuple: (data_matrix, wavelengths, timestamps)
        """
        # Convert bytes/memoryview to string
        if isinstance(file_content, bytes):
            content_str = file_content.decode('utf-8')
        elif isinstance(file_content, memoryview):
            content_str = file_content.tobytes().decode('utf-8')
        else:
            content_str = str(file_content)
            
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
        self.timestamps = np.array([float(x) for x in valid_timestamps])
        
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
                intensity_strings = parts[1:len(self.timestamps)+1]
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

    def load_h5_data(self, mode, h5_path):
        print("Start of load_h5_data")
        import h5py
        with h5py.File(h5_path, "r") as f:
            if mode == "pl_raw":
                timestamps = f["/raw_optical_measurements/raw_pl_measurements/raw_pl_Time"][()]
                data_matrix = f["/raw_optical_measurements/raw_pl_measurements/raw_pl_data"][()].T
                wavelengths = f["/raw_optical_measurements/wavelengths_spectrometer/wavelengths_spectrometer_data"][
                    ()]
            if mode == "pl_binned":
                extent = f["/binned_optical_measurements/time_extent_for_binning"][()]
                data_matrix = f["/binned_optical_measurements/binned_pl_measurements_bg"][()]
                timestamps, wavelengths = get_axes_from_extent(extent, data_matrix)
            if mode == "giwaxs":
                timestamps = f["/beamline_logging/Time"][()]
                data_matrix = f["/diffractogram/i_values"][()].T
                wavelengths = f["/diffractogram/q_values"][()][0]

        return data_matrix, wavelengths, timestamps
