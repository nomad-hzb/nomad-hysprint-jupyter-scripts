"""
Utilities for the Design of Experiments application.
Helper functions, validation utilities, and constants.
"""

import numpy as np
import pandas as pd
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import warnings


class Constants:
    """Application constants and configuration."""
    
    APPLICATION_VERSION = "1.0.0"
    APPLICATION_NAME = "Design of Experiments (DoE) Application"
    
    # Sampling limits
    MIN_SAMPLES = 4
    MAX_SAMPLES = 10000
    DEFAULT_SAMPLES = 25
    
    # Variable limits
    MAX_VARIABLES = 20
    MAX_CATEGORIES = 50
    
    # File size limits (in bytes)
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Supported file formats
    SUPPORTED_IMPORT_FORMATS = ['.json', '.csv', '.xlsx']
    SUPPORTED_EXPORT_FORMATS = ['csv', 'json', 'excel']
    
    # Algorithm requirements
    ALGORITHM_REQUIREMENTS = {
        "Latin Hypercube Sampling": [],
        "Sobol Sequences": [],
        "Halton Sequences": [],
        "Random Sampling": [],
        "Uniform Grid Sampling": [],
        "Orthogonal Arrays": ["pyDOE2"],
        "Maximin Distance Design": ["scikit-learn"]
    }
    
    # Quality metric thresholds
    QUALITY_THRESHOLDS = {
        'min_distance': {'good': 0.1, 'fair': 0.05},
        'overall_coverage': {'good': 0.8, 'fair': 0.6},
        'distance_uniformity': {'good': 0.3, 'fair': 0.5},
        'max_correlation': {'good': 0.3, 'fair': 0.5}
    }


class ValidationUtils:
    """Utility class for data validation and error checking."""
    
    @staticmethod
    def validate_variable_name(name: str) -> Tuple[bool, str]:
        """Validate variable name format and content."""
        if not name or not name.strip():
            return False, "Variable name cannot be empty"
        
        name = name.strip()
        
        # Check length
        if len(name) > 50:
            return False, "Variable name too long (max 50 characters)"
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
            return False, "Variable name must start with letter and contain only letters, numbers, underscore, or hyphen"
        
        # Check for reserved words
        reserved_words = ['experiment_id', 'sample_id', 'index', 'id']
        if name.lower() in reserved_words:
            return False, f"'{name}' is a reserved word"
        
        return True, ""
    
    @staticmethod
    def validate_numeric_range(min_val: float, max_val: float, 
                              step: Optional[float] = None) -> Tuple[bool, str]:
        """Validate numeric range parameters."""
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            return False, "Min and max values must be numeric"
        
        if np.isnan(min_val) or np.isnan(max_val):
            return False, "Min and max values cannot be NaN"
        
        if np.isinf(min_val) or np.isinf(max_val):
            return False, "Min and max values cannot be infinite"
        
        if min_val >= max_val:
            return False, "Max value must be greater than min value"
        
        if step is not None:
            if not isinstance(step, (int, float)):
                return False, "Step size must be numeric"
            
            if step <= 0:
                return False, "Step size must be positive"
            
            if step >= (max_val - min_val):
                return False, "Step size must be smaller than the range"
        
        return True, ""
    
    @staticmethod
    def validate_categories(categories: List[str]) -> Tuple[bool, str]:
        """Validate categorical variable categories."""
        if not categories:
            return False, "At least one category is required"
        
        if len(categories) < 2:
            return False, "At least two categories are required"
        
        if len(categories) > Constants.MAX_CATEGORIES:
            return False, f"Too many categories (max {Constants.MAX_CATEGORIES})"
        
        # Check for empty or whitespace-only categories
        valid_categories = [cat.strip() for cat in categories if cat.strip()]
        if len(valid_categories) != len(categories):
            return False, "Categories cannot be empty or whitespace-only"
        
        # Check for duplicates
        if len(set(valid_categories)) != len(valid_categories):
            return False, "Duplicate categories are not allowed"
        
        # Check category length
        for cat in valid_categories:
            if len(cat) > 100:
                return False, f"Category '{cat}' is too long (max 100 characters)"
        
        return True, ""
    
    @staticmethod
    def validate_sample_size(n_samples: int, n_variables: int) -> Tuple[bool, str]:
        """Validate sample size given number of variables."""
        if not isinstance(n_samples, int):
            return False, "Sample size must be an integer"
        
        if n_samples < Constants.MIN_SAMPLES:
            return False, f"Sample size must be at least {Constants.MIN_SAMPLES}"
        
        if n_samples > Constants.MAX_SAMPLES:
            return False, f"Sample size cannot exceed {Constants.MAX_SAMPLES}"
        
        # Check recommended minimum
        recommended_min = max(n_variables ** 2, Constants.MIN_SAMPLES)
        if n_samples < recommended_min:
            return True, f"Warning: Recommended minimum is {recommended_min} samples for {n_variables} variables"
        
        return True, ""
    
    @staticmethod
    def validate_json_structure(json_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate JSON configuration structure."""
        required_fields = ['variables']
        
        for field in required_fields:
            if field not in json_data:
                return False, f"Missing required field: {field}"
        
        # Validate variables array
        variables = json_data['variables']
        if not isinstance(variables, list):
            return False, "'variables' must be an array"
        
        if len(variables) == 0:
            return False, "At least one variable is required"
        
        if len(variables) > Constants.MAX_VARIABLES:
            return False, f"Too many variables (max {Constants.MAX_VARIABLES})"
        
        # Validate each variable
        required_var_fields = ['name', 'type']
        for i, var in enumerate(variables):
            if not isinstance(var, dict):
                return False, f"Variable {i+1} must be an object"
            
            for field in required_var_fields:
                if field not in var:
                    return False, f"Variable {i+1} missing required field: {field}"
            
            # Validate variable name
            is_valid, error_msg = ValidationUtils.validate_variable_name(var['name'])
            if not is_valid:
                return False, f"Variable {i+1}: {error_msg}"
        
        return True, ""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe downloads."""
        # Remove or replace unsafe characters
        safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
        
        # Limit length
        if len(safe_filename) > 100:
            name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
            safe_filename = name[:95] + ('.' + ext if ext else '')
        
        return safe_filename
    
    @staticmethod
    def format_number(value: Union[int, float], decimals: int = 3) -> str:
        """Format number for display with appropriate precision."""
        if pd.isna(value):
            return "N/A"
        
        if isinstance(value, int) or value == int(value):
            return str(int(value))
        
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def calculate_file_size(content: str) -> str:
        """Calculate and format file size."""
        size_bytes = len(content.encode('utf-8'))
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.1f} MB"
        else:
            return f"{size_bytes/1024**3:.1f} GB"


class DataProcessor:
    """Utility class for data processing and transformation."""
    
    @staticmethod
    def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize data using specified method."""
        if data.empty:
            return data
        
        result = data.copy()
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            for col in numeric_columns:
                col_min, col_max = result[col].min(), result[col].max()
                if col_max > col_min:
                    result[col] = (result[col] - col_min) / (col_max - col_min)
        
        elif method == 'zscore':
            # Z-score standardization
            for col in numeric_columns:
                col_mean, col_std = result[col].mean(), result[col].std()
                if col_std > 0:
                    result[col] = (result[col] - col_mean) / col_std
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            for col in numeric_columns:
                col_median = result[col].median()
                col_q75 = result[col].quantile(0.75)
                col_q25 = result[col].quantile(0.25)
                col_iqr = col_q75 - col_q25
                if col_iqr > 0:
                    result[col] = (result[col] - col_median) / col_iqr
        
        return result
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 1.5) -> Dict[str, List[int]]:
        """Detect outliers in numeric columns."""
        outliers = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            col_outliers = []
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = data.index[
                    (data[col] < lower_bound) | (data[col] > upper_bound)
                ].tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                col_outliers = data.index[z_scores > threshold].tolist()
            
            elif method == 'modified_zscore':
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                if mad > 0:
                    modified_z_scores = 0.6745 * (col_data - median) / mad
                    col_outliers = data.index[np.abs(modified_z_scores) > threshold].tolist()
            
            if col_outliers:
                outliers[col] = col_outliers
        
        return outliers
    
    @staticmethod
    def calculate_summary_stats(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive summary statistics."""
        summary = {}
        
        for col in data.columns:
            col_data = data[col]
            col_summary = {}
            
            if col_data.dtype in ['object', 'category']:
                # Categorical statistics
                col_summary.update({
                    'count': len(col_data),
                    'unique': col_data.nunique(),
                    'top': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
                    'freq': col_data.value_counts().iloc[0] if len(col_data) > 0 else 0,
                    'missing': col_data.isnull().sum()
                })
            else:
                # Numeric statistics
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                col_summary.update({
                    'count': len(numeric_data.dropna()),
                    'mean': numeric_data.mean(),
                    'std': numeric_data.std(),
                    'min': numeric_data.min(),
                    'q25': numeric_data.quantile(0.25),
                    'median': numeric_data.median(),
                    'q75': numeric_data.quantile(0.75),
                    'max': numeric_data.max(),
                    'missing': numeric_data.isnull().sum(),
                    'skewness': numeric_data.skew(),
                    'kurtosis': numeric_data.kurtosis()
                })
            
            summary[col] = col_summary
        
        return summary


class ExperimentalDesignUtils:
    """Utilities specific to experimental design and DoE."""
    
    @staticmethod
    def calculate_design_efficiency(samples: pd.DataFrame, variables: List, 
                                  algorithm: str) -> Dict[str, float]:
        """Calculate design efficiency metrics."""
        if samples.empty or not variables:
            return {}
        
        n_samples = len(samples)
        n_vars = len(variables)
        
        efficiency = {}
        
        # Sample efficiency (samples per variable)
        efficiency['samples_per_variable'] = n_samples / n_vars if n_vars > 0 else 0
        
        # Coverage efficiency (design space utilization)
        total_coverage = 1.0
        for var in variables:
            if var.name not in samples.columns:
                continue
            
            if hasattr(var, 'type'):
                from data_manager import VariableType
                if var.type == VariableType.CONTINUOUS:
                    data_range = samples[var.name].max() - samples[var.name].min()
                    total_range = var.max_value - var.min_value
                    coverage = data_range / total_range if total_range > 0 else 1.0
                    total_coverage *= coverage
        
        efficiency['coverage_efficiency'] = total_coverage
        
        # Orthogonality (for applicable algorithms)
        if algorithm in ["Orthogonal Arrays", "Latin Hypercube Sampling"]:
            numeric_data = samples.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                # Get upper triangle correlations (excluding diagonal)
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                avg_correlation = upper_tri.stack().abs().mean()
                efficiency['orthogonality'] = 1.0 - avg_correlation
        
        return efficiency
    
    @staticmethod
    def recommend_sample_size(n_variables: int, algorithm: str, 
                            objectives: List[str] = None) -> Dict[str, int]:
        """Recommend sample sizes based on experimental objectives."""
        recommendations = {}
        
        # Base recommendation (rule of thumb)
        base_recommendation = max(n_variables ** 2, Constants.MIN_SAMPLES)
        recommendations['minimum'] = base_recommendation
        
        # Algorithm-specific adjustments
        if algorithm == "Latin Hypercube Sampling":
            recommendations['recommended'] = max(base_recommendation, n_variables * 10)
            recommendations['optimal'] = n_variables * 20
        
        elif algorithm in ["Sobol Sequences", "Halton Sequences"]:
            # Quasi-random sequences benefit from power-of-2 sizes
            power_of_2 = 2 ** int(np.ceil(np.log2(base_recommendation)))
            recommendations['recommended'] = power_of_2
            recommendations['optimal'] = power_of_2 * 2
        
        elif algorithm == "Orthogonal Arrays":
            # OA sizes depend on available arrays
            recommendations['recommended'] = base_recommendation
            recommendations['optimal'] = base_recommendation * 2
        
        else:
            recommendations['recommended'] = base_recommendation * 2
            recommendations['optimal'] = base_recommendation * 4
        
        # Objective-specific adjustments
        if objectives:
            if 'screening' in objectives:
                # For screening, fewer samples may be acceptable
                recommendations['screening'] = max(n_variables * 2, Constants.MIN_SAMPLES)
            
            if 'optimization' in objectives:
                # For optimization, more samples are typically needed
                recommendations['optimization'] = n_variables * 50
            
            if 'modeling' in objectives:
                # For response surface modeling
                recommendations['modeling'] = n_variables * 30
        
        return recommendations
    
    @staticmethod
    def assess_design_quality(samples: pd.DataFrame, variables: List,
                            metrics: Dict[str, float]) -> Dict[str, str]:
        """Assess overall design quality and provide recommendations."""
        if not metrics:
            return {'overall': 'unknown', 'recommendations': []}
        
        quality_scores = {}
        recommendations = []
        
        # Assess individual metrics
        for metric, value in metrics.items():
            if metric in Constants.QUALITY_THRESHOLDS:
                thresholds = Constants.QUALITY_THRESHOLDS[metric]
                
                if metric in ['min_distance', 'overall_coverage']:
                    # Higher is better
                    if value >= thresholds['good']:
                        quality_scores[metric] = 'good'
                    elif value >= thresholds['fair']:
                        quality_scores[metric] = 'fair'
                    else:
                        quality_scores[metric] = 'poor'
                        if metric == 'min_distance':
                            recommendations.append("Consider increasing sample size or using a different algorithm")
                        elif metric == 'overall_coverage':
                            recommendations.append("Consider adjusting variable ranges or sample distribution")
                
                else:
                    # Lower is better (e.g., correlation, uniformity measures)
                    if value <= thresholds['good']:
                        quality_scores[metric] = 'good'
                    elif value <= thresholds['fair']:
                        quality_scores[metric] = 'fair'
                    else:
                        quality_scores[metric] = 'poor'
                        if metric == 'distance_uniformity':
                            recommendations.append("Sample distribution is not uniform - consider different algorithm")
                        elif metric == 'max_correlation':
                            recommendations.append("High correlation between variables - check for redundancy")
        
        # Overall assessment
        if not quality_scores:
            overall_quality = 'unknown'
        else:
            good_count = sum(1 for q in quality_scores.values() if q == 'good')
            poor_count = sum(1 for q in quality_scores.values() if q == 'poor')
            
            if poor_count == 0 and good_count >= len(quality_scores) * 0.7:
                overall_quality = 'excellent'
            elif poor_count == 0:
                overall_quality = 'good'
            elif poor_count <= len(quality_scores) * 0.3:
                overall_quality = 'fair'
            else:
                overall_quality = 'poor'
                recommendations.append("Consider regenerating samples with different parameters")
        
        return {
            'overall': overall_quality,
            'individual_scores': quality_scores,
            'recommendations': recommendations
        }


class FileHandler:
    """Utility class for file handling operations."""
    
    @staticmethod
    def create_csv_content(data: pd.DataFrame, include_index: bool = False) -> str:
        """Create CSV content from DataFrame."""
        return data.to_csv(index=include_index, encoding='utf-8')
    
    @staticmethod
    def create_json_content(data: Dict[str, Any], indent: int = 2) -> str:
        """Create JSON content from dictionary."""
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    
    @staticmethod
    def create_excel_content(data_dict: Dict[str, pd.DataFrame]) -> bytes:
        """Create Excel content with multiple sheets."""
        try:
            import io
            from openpyxl import Workbook
            from openpyxl.utils.dataframe import dataframe_to_rows
            
            wb = Workbook()
            
            # Remove default sheet
            wb.remove(wb.active)
            
            for sheet_name, df in data_dict.items():
                ws = wb.create_sheet(title=sheet_name)
                
                # Add DataFrame to worksheet
                for r in dataframe_to_rows(df, index=False, header=True):
                    ws.append(r)
            
            # Save to bytes
            excel_buffer = io.BytesIO()
            wb.save(excel_buffer)
            excel_buffer.seek(0)
            
            return excel_buffer.getvalue()
        
        except ImportError:
            warnings.warn("openpyxl not available for Excel export")
            return b""
    
    @staticmethod
    def parse_uploaded_file(file_content: bytes, filename: str) -> Tuple[bool, str, Any]:
        """Parse uploaded file content based on extension."""
        try:
            if filename.endswith('.json'):
                content_str = file_content.decode('utf-8')
                data = json.loads(content_str)
                return True, "JSON file parsed successfully", data
            
            elif filename.endswith('.csv'):
                content_str = file_content.decode('utf-8')
                # Try to detect delimiter
                import csv
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(content_str[:1024])
                    delimiter = dialect.delimiter
                except:
                    delimiter = ','
                
                df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)
                return True, "CSV file parsed successfully", df
            
            elif filename.endswith(('.xlsx', '.xls')):
                try:
                    df = pd.read_excel(io.BytesIO(file_content))
                    return True, "Excel file parsed successfully", df
                except ImportError:
                    return False, "openpyxl not available for Excel import", None
            
            else:
                return False, f"Unsupported file format: {filename}", None
        
        except Exception as e:
            return False, f"Error parsing file: {str(e)}", None


class ColorUtils:
    """Utility class for color management and palette generation."""
    
    @staticmethod
    def generate_color_palette(n_colors: int, palette_type: str = 'qualitative') -> List[str]:
        """Generate color palette with specified number of colors."""
        try:
            import plotly.colors as pc
            
            if palette_type == 'qualitative':
                base_colors = pc.qualitative.Set3
            elif palette_type == 'sequential':
                base_colors = pc.sequential.Viridis
            elif palette_type == 'diverging':
                base_colors = pc.diverging.RdBu
            else:
                base_colors = pc.qualitative.Plotly
            
            # Repeat colors if needed
            colors = []
            for i in range(n_colors):
                colors.append(base_colors[i % len(base_colors)])
            
            return colors
        
        except ImportError:
            # Fallback colors if plotly not available
            fallback_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            return [fallback_colors[i % len(fallback_colors)] for i in range(n_colors)]
    
    @staticmethod
    def color_by_quality(value: float, metric_type: str = 'higher_better') -> str:
        """Return color based on quality assessment."""
        if pd.isna(value):
            return '#cccccc'  # Gray for missing values
        
        if metric_type == 'higher_better':
            if value >= 0.8:
                return '#2ca02c'  # Green for good
            elif value >= 0.6:
                return '#ff7f0e'  # Orange for fair
            else:
                return '#d62728'  # Red for poor
        else:  # lower_better
            if value <= 0.2:
                return '#2ca02c'  # Green for good
            elif value <= 0.4:
                return '#ff7f0e'  # Orange for fair
            else:
                return '#d62728'  # Red for poor


class MathUtils:
    """Mathematical utility functions for DoE calculations."""
    
    @staticmethod
    def calculate_distance_matrix(points: np.ndarray) -> np.ndarray:
        """Calculate full distance matrix between all points."""
        from scipy.spatial.distance import cdist
        return cdist(points, points)
    
    @staticmethod
    def calculate_volume_coverage(points: np.ndarray) -> float:
        """Calculate approximate volume coverage in unit hypercube."""
        if len(points) == 0:
            return 0.0
        
        n_points, n_dims = points.shape
        
        # Simple approximation: ratio of convex hull volume to unit hypercube
        try:
            from scipy.spatial import ConvexHull
            
            if n_points >= n_dims + 1:
                hull = ConvexHull(points)
                hull_volume = hull.volume
                unit_volume = 1.0  # Unit hypercube volume
                return min(hull_volume / unit_volume, 1.0)
            else:
                # Not enough points for convex hull
                return 0.0
        
        except ImportError:
            # Fallback: range-based approximation
            ranges = []
            for dim in range(n_dims):
                dim_range = points[:, dim].max() - points[:, dim].min()
                ranges.append(dim_range)
            
            return np.prod(ranges)
    
    @staticmethod
    def calculate_discrepancy(points: np.ndarray, method: str = 'star') -> float:
        """Calculate discrepancy measure for space-filling quality."""
        if len(points) == 0:
            return float('inf')
        
        n_points, n_dims = points.shape
        
        if method == 'star':
            # Star discrepancy (simplified calculation)
            max_discrepancy = 0.0
            
            # Sample a subset of test points for efficiency
            test_points = np.random.random((min(100, n_points), n_dims))
            
            for test_point in test_points:
                # Count points in the box [0, test_point]
                in_box = np.all(points <= test_point, axis=1).sum()
                empirical_measure = in_box / n_points
                
                # Theoretical measure (volume of box)
                theoretical_measure = np.prod(test_point)
                
                discrepancy = abs(empirical_measure - theoretical_measure)
                max_discrepancy = max(max_discrepancy, discrepancy)
            
            return max_discrepancy
        
        else:
            return 0.0
    
    @staticmethod
    def calculate_condition_number(matrix: np.ndarray) -> float:
        """Calculate condition number of a matrix."""
        try:
            return np.linalg.cond(matrix)
        except np.linalg.LinAlgError:
            return float('inf')
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, 
                   default: float = 0.0) -> float:
        """Safe division with default value for division by zero."""
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator


# Convenience functions for common operations
def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate string to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def generate_experiment_id() -> str:
    """Generate unique experiment identifier."""
    import uuid
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"