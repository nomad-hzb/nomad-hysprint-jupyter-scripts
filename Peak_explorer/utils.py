# utils.py
"""
Utility functions for Photoluminescence Analysis App
"""
import config
from datetime import datetime


def debug_print(message, category="DEBUG"):
    """
    Print debug messages if DEBUG_MODE is enabled
    
    Parameters:
    -----------
    message : str
        Debug message to print
    category : str, optional
        Category/type of debug message (e.g., "DATA", "FITTING", "GUI")
    """
    if config.DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{category}] {message}")


def format_timestamp(timestamp_format=None):
    """
    Get formatted timestamp string
    
    Parameters:
    -----------
    timestamp_format : str, optional
        Format string for datetime. If None, uses config default
        
    Returns:
    --------
    str: Formatted timestamp
    """
    if timestamp_format is None:
        timestamp_format = config.TIMESTAMP_FORMAT
    return datetime.now().strftime(timestamp_format)


def validate_time_index(time_idx, max_idx):
    """
    Validate and clip time index to valid range
    
    Parameters:
    -----------
    time_idx : int
        Time index to validate
    max_idx : int
        Maximum valid index
        
    Returns:
    --------
    int: Valid time index
    """
    return max(0, min(time_idx, max_idx))


def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, returning default if division fails
    
    Parameters:
    -----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float, optional
        Default value if division fails
        
    Returns:
    --------
    float: Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def generate_output_filename(prefix, extension, include_timestamp=True):
    """
    Generate output filename with optional timestamp
    
    Parameters:
    -----------
    prefix : str
        Filename prefix
    extension : str
        File extension (with or without dot)
    include_timestamp : bool, optional
        Whether to include timestamp in filename
        
    Returns:
    --------
    str: Generated filename
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    if include_timestamp:
        timestamp = format_timestamp()
        return f"{prefix}_{timestamp}{extension}"
    else:
        return f"{prefix}{extension}"
