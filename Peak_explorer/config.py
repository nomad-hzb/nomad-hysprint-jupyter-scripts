# config.py
"""
Configuration settings for Photoluminescence Analysis App
"""

# =============================================================================
# DEBUG SETTINGS
# =============================================================================
DEBUG_MODE = False  # Set to True to enable debug output throughout the app

# =============================================================================
# UI SETTINGS
# =============================================================================
# Widget dimensions
CONTROL_PANEL_WIDTH = '420px'
TIME_SLIDER_WIDTH = '300px'
TIME_INPUT_WIDTH = '150px'

# Plot dimensions
HEATMAP_HEIGHT = 500
HEATMAP_WIDTH = 800
SPECTRUM_HEIGHT = 600
SPECTRUM_WIDTH = 800

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
# Colorscales
DEFAULT_COLORSCALE = 'Viridis'

# Plot rendering
PLOT_RENDERER = 'jupyterlab'  # Options: 'svg', 'png', 'notebook'

# =============================================================================
# FITTING SETTINGS
# =============================================================================
# Peak detection defaults
PEAK_DETECTION_DEFAULTS = {
    'height': None,
    'threshold': None,
    'distance': 5,
    'prominence': None,
    'width': None,
    'wlen': None,
    'rel_height': 0.5,
    'plateau_size': None
}

# Fitting defaults
DEFAULT_BACKGROUND_MODEL = 'Linear'
DEFAULT_PEAK_MODEL = 'Gaussian'

# Background handling
BACKGROUND_OPTIONS = ['None', 'Manual', 'Linear', 'Polynomial', 'Exponential', 'Custom']
DEFAULT_BACKGROUND_METHOD = 'None'
DEFAULT_BACKGROUND_START_IDX = 0
DEFAULT_BACKGROUND_NUM_CURVES = 10
DEFAULT_POLY_DEGREE = 2

# Parallel processing
USE_PARALLEL_FITTING = True
MAX_WORKERS = None  # None = auto-detect based on CPU count

# Smart initialization for batch fitting
USE_SMART_INIT = True
SMART_INIT_SEARCH_RADIUS = 5

# =============================================================================
# EXPORT SETTINGS
# =============================================================================
# Default export formats
EXPORT_FORMATS = ['xlsx', 'csv', 'json', 'hdf5']
DEFAULT_EXPORT_FORMAT = 'xlsx'

# Output directory naming
OUTPUT_DIR_PREFIX = 'pl_analysis_results'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# =============================================================================
# H5 FILE SETTINGS
# =============================================================================
# H5 data modes
H5_MODES = {
    'pl_raw': 'PL raw',
    'pl_binned': 'PL binned & bgs',
    'giwaxs': 'GIWAXS',
    'transmission_raw': 'Transmission raw',
    'transmission_binned': 'Transmission binned & bgs'
}
DEFAULT_H5_MODE = 'giwaxs'

# H5 paths (dataset locations within H5 file)
H5_PATHS = {
    'pl_raw': {
        'timestamps': '/raw_optical_measurements/raw_pl_measurements/raw_pl_Time',
        'data': '/raw_optical_measurements/raw_pl_measurements/raw_pl_data',
        'wavelengths': '/raw_optical_measurements/wavelengths_spectrometer/wavelengths_spectrometer_data'
    },
    'pl_binned': {
        'extent': '/binned_optical_measurements/time_extent_for_binning',
        'data': '/binned_optical_measurements/binned_pl_measurements_bg'
    },
    'giwaxs': {
        'timestamps': '/beamline_logging/Time',
        'data': '/diffractogram/i_values',
        'wavelengths': '/diffractogram/q_values'
    },
    'transmission_raw': {
        'timestamps': '/raw_optical_measurements/raw_transmission_measurements/raw_transmission_Time',
        'data': '/raw_optical_measurements/raw_transmission_measurements/raw_transmission_data',
        'wavelengths': '/raw_optical_measurements/wavelengths_spectrometer/wavelengths_spectrometer_data'
    },
    'transmission_binned': {
        'extent': '/binned_optical_measurements/time_extent_for_binning',
        'data': '/binned_optical_measurements/binned_transmission_measurements_bg'
    }
}

# =============================================================================
# FILE UPLOAD SETTINGS
# =============================================================================
ACCEPTED_FILE_TYPES = '.txt,.csv,.dat'
ACCEPT_MULTIPLE_FILES = False

# =============================================================================
# DATA VALIDATION
# =============================================================================
CHECK_FOR_NAN = True
CHECK_FOR_NEGATIVE = True
CHECK_WAVELENGTH_ORDER = True
CHECK_TIME_ORDER = True
