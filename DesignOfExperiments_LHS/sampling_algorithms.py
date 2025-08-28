"""
Sampling Algorithms for the Design of Experiments application.
Implementation of all DoE sampling algorithms with quality metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Scientific computing imports
from scipy.stats import qmc
from scipy.spatial.distance import pdist, squareform
from scipy.stats import uniform, randint
from sklearn.preprocessing import MinMaxScaler

# Optional imports for advanced algorithms
try:
    import pyDOE2
    HAS_PYDOE2 = True
except ImportError:
    HAS_PYDOE2 = False
    warnings.warn("pyDOE2 not available. Some algorithms will be unavailable.")

try:
    from skopt.sampler import Lhs, Sobol, Halton
    from skopt.space import Real, Integer, Categorical
    HAS_SCIKIT_OPTIMIZE = True
except ImportError:
    HAS_SCIKIT_OPTIMIZE = False
    warnings.warn("scikit-optimize not available. Some advanced algorithms will be unavailable.")

from data_manager import Variable, VariableType
from utils import ValidationUtils


class SamplingAlgorithm(ABC):
    """Abstract base class for sampling algorithms."""
    
    def __init__(self, name: str, description: str, requires_libraries: List[str] = None):
        self.name = name
        self.description = description
        self.requires_libraries = requires_libraries or []
    
    @abstractmethod
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate samples for given variables."""
        pass
    
    def is_available(self) -> bool:
        """Check if the algorithm is available (dependencies satisfied)."""
        return True  # Override in subclasses that require specific libraries
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm-specific parameters and their default values."""
        return {}


class LatinHypercubeSampling(SamplingAlgorithm):
    """Latin Hypercube Sampling implementation."""
    
    def __init__(self):
        super().__init__(
            name="Latin Hypercube Sampling",
            description="Space-filling design with one sample in each row and column projection"
        )
    
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate LHS samples."""
        # Get optimization parameter
        optimization = kwargs.get('optimization', None)
        
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=len(variables), seed=random_state, optimization=optimization)
        
        # Generate unit hypercube samples
        unit_samples = sampler.random(n=n_samples)
        
        # Transform to variable spaces
        samples = self._transform_to_variable_space(unit_samples, variables)
        
        return pd.DataFrame(samples, columns=[var.name for var in variables])
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get LHS-specific parameters."""
        return {
            'optimization': {
                'type': 'select',
                'options': [None, 'random-cd', 'lloyd'],
                'default': None,
                'description': 'Optimization method for LHS'
            }
        }
    
    def _transform_to_variable_space(self, unit_samples: np.ndarray, 
                                   variables: List[Variable]) -> np.ndarray:
        """Transform unit hypercube samples to variable spaces."""
        n_samples, n_vars = unit_samples.shape
        samples = []  # Use list instead of numpy array
        
        for i, var in enumerate(variables):
            if var.type == VariableType.CONTINUOUS:
                var_samples = var.min_value + unit_samples[:, i] * (var.max_value - var.min_value)
                samples.append(var_samples)
            
            elif var.type == VariableType.DISCRETE:
                discrete_values = var.get_discrete_values()
                indices = (unit_samples[:, i] * len(discrete_values)).astype(int)
                indices = np.clip(indices, 0, len(discrete_values) - 1)
                var_samples = [discrete_values[idx] for idx in indices]
                samples.append(var_samples)
            
            elif var.type == VariableType.CATEGORICAL:
                indices = (unit_samples[:, i] * len(var.categories)).astype(int)
                indices = np.clip(indices, 0, len(var.categories) - 1)
                var_samples = [var.categories[idx] for idx in indices]
                samples.append(var_samples)
        
        # Convert to DataFrame-compatible format
        return np.column_stack(samples)


class SobolSampling(SamplingAlgorithm):
    """Sobol sequence sampling implementation."""
    
    def __init__(self):
        super().__init__(
            name="Sobol Sequences",
            description="Low-discrepancy quasi-random sequences with excellent space-filling properties"
        )
    
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate Sobol sequence samples."""
        scramble = kwargs.get('scramble', True)
        
        # Create Sobol sampler
        sampler = qmc.Sobol(d=len(variables), scramble=scramble, seed=random_state)
        
        # Generate unit hypercube samples
        unit_samples = sampler.random(n=n_samples)
        
        # Transform to variable spaces
        samples = LatinHypercubeSampling()._transform_to_variable_space(unit_samples, variables)
        
        return pd.DataFrame(samples, columns=[var.name for var in variables])
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Sobol-specific parameters."""
        return {
            'scramble': {
                'type': 'boolean',
                'default': True,
                'description': 'Apply Owen scrambling to improve randomization'
            }
        }


class HaltonSampling(SamplingAlgorithm):
    """Halton sequence sampling implementation."""
    
    def __init__(self):
        super().__init__(
            name="Halton Sequences",
            description="Quasi-random sequences with good space-filling properties"
        )
    
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate Halton sequence samples."""
        scramble = kwargs.get('scramble', True)
        
        # Create Halton sampler
        sampler = qmc.Halton(d=len(variables), scramble=scramble, seed=random_state)
        
        # Skip initial samples to avoid low-discrepancy issues
        skip_samples = kwargs.get('skip_samples', 100)
        if skip_samples > 0:
            _ = sampler.random(n=skip_samples)
        
        # Generate unit hypercube samples
        unit_samples = sampler.random(n=n_samples)
        
        # Transform to variable spaces
        samples = LatinHypercubeSampling()._transform_to_variable_space(unit_samples, variables)
        
        return pd.DataFrame(samples, columns=[var.name for var in variables])
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Halton-specific parameters."""
        return {
            'scramble': {
                'type': 'boolean',
                'default': True,
                'description': 'Apply scrambling to improve randomization'
            },
            'skip_samples': {
                'type': 'integer',
                'default': 100,
                'min': 0,
                'max': 1000,
                'description': 'Number of initial samples to skip'
            }
        }


class RandomSampling(SamplingAlgorithm):
    """Simple random sampling implementation."""
    
    def __init__(self):
        super().__init__(
            name="Random Sampling",
            description="Simple random sampling for baseline comparison"
        )
    
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate random samples."""
        rng = np.random.RandomState(random_state)
        samples = []
        
        for var in variables:
            if var.type == VariableType.CONTINUOUS:
                var_samples = rng.uniform(var.min_value, var.max_value, n_samples)
            
            elif var.type == VariableType.DISCRETE:
                discrete_values = var.get_discrete_values()
                var_samples = rng.choice(discrete_values, n_samples)
            
            elif var.type == VariableType.CATEGORICAL:
                var_samples = rng.choice(var.categories, n_samples)
            
            samples.append(var_samples)
        
        samples_array = np.column_stack(samples)
        return pd.DataFrame(samples_array, columns=[var.name for var in variables])


class UniformGridSampling(SamplingAlgorithm):
    """Uniform grid sampling implementation."""
    
    def __init__(self):
        super().__init__(
            name="Uniform Grid Sampling",
            description="Regular grid sampling for systematic coverage"
        )
    
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate uniform grid samples."""
        # Calculate grid dimensions
        n_dims = len(variables)
        samples_per_dim = int(np.ceil(n_samples ** (1/n_dims)))
        
        # Generate grid points for each variable
        grid_points = []
        for var in variables:
            if var.type == VariableType.CONTINUOUS:
                points = np.linspace(var.min_value, var.max_value, samples_per_dim)
            
            elif var.type == VariableType.DISCRETE:
                discrete_values = var.get_discrete_values()
                # Sample evenly from discrete values
                indices = np.linspace(0, len(discrete_values)-1, samples_per_dim).astype(int)
                points = [discrete_values[i] for i in indices]
            
            elif var.type == VariableType.CATEGORICAL:
                # Repeat categories to fill grid
                points = (var.categories * (samples_per_dim // len(var.categories) + 1))[:samples_per_dim]
            
            grid_points.append(points)
        
        # Create full factorial grid
        grid_mesh = np.array(np.meshgrid(*grid_points, indexing='ij'))
        grid_samples = grid_mesh.reshape(n_dims, -1).T
        
        # Randomly sample from grid if we have too many points
        if len(grid_samples) > n_samples:
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(grid_samples), n_samples, replace=False)
            grid_samples = grid_samples[indices]
        
        return pd.DataFrame(grid_samples, columns=[var.name for var in variables])


class OrthogonalArraySampling(SamplingAlgorithm):
    """Orthogonal Array sampling implementation."""
    
    def __init__(self):
        super().__init__(
            name="Orthogonal Arrays",
            description="Classical DoE approach excellent for screening experiments",
            requires_libraries=['pyDOE2']
        )
    
    def is_available(self) -> bool:
        """Check if pyDOE2 is available."""
        return HAS_PYDOE2
    
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate orthogonal array samples."""
        if not self.is_available():
            raise ImportError("pyDOE2 is required for Orthogonal Array sampling")
        
        # Determine levels for each variable
        levels = []
        for var in variables:
            if var.type == VariableType.CATEGORICAL:
                levels.append(len(var.categories))
            elif var.type == VariableType.DISCRETE:
                levels.append(len(var.get_discrete_values()))
            else:  # CONTINUOUS - discretize
                levels.append(kwargs.get('continuous_levels', 5))
        
        # Try to find suitable orthogonal array
        try:
            # Use pyDOE2 to generate orthogonal array
            oa_samples = pyDOE2.gsd(levels, n_samples)
            
            # Transform to actual variable values
            samples = []
            for i, var in enumerate(variables):
                if var.type == VariableType.CONTINUOUS:
                    # Map orthogonal array levels to continuous range
                    n_levels = levels[i]
                    level_values = np.linspace(var.min_value, var.max_value, n_levels)
                    var_samples = [level_values[int(level)] for level in oa_samples[:, i]]
                
                elif var.type == VariableType.DISCRETE:
                    discrete_values = var.get_discrete_values()
                    var_samples = [discrete_values[int(level)] for level in oa_samples[:, i]]
                
                elif var.type == VariableType.CATEGORICAL:
                    var_samples = [var.categories[int(level)] for level in oa_samples[:, i]]
                
                samples.append(var_samples)
            
            samples_array = np.column_stack(samples)
            return pd.DataFrame(samples_array, columns=[var.name for var in variables])
            
        except Exception as e:
            # Fallback to random sampling if orthogonal array generation fails
            warnings.warn(f"Orthogonal array generation failed: {e}. Using random sampling.")
            return RandomSampling().generate(variables, n_samples, random_state, **kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Orthogonal Array specific parameters."""
        return {
            'continuous_levels': {
                'type': 'integer',
                'default': 5,
                'min': 2,
                'max': 10,
                'description': 'Number of levels for continuous variables'
            }
        }


class MaximinDistanceSampling(SamplingAlgorithm):
    """Maximin distance design implementation."""
    
    def __init__(self):
        super().__init__(
            name="Maximin Distance Design",
            description="Optimizes minimum distance between sample points"
        )
    
    def generate(self, variables: List[Variable], n_samples: int, 
                random_state: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Generate maximin distance samples."""
        max_iterations = kwargs.get('max_iterations', 100)
        n_candidates = kwargs.get('n_candidates', n_samples * 10)
        
        # Generate candidate samples using LHS
        lhs_sampler = LatinHypercubeSampling()
        candidates = lhs_sampler.generate(variables, n_candidates, random_state)
        
        # Convert to numeric array for distance calculations
        numeric_candidates = self._convert_to_numeric(candidates, variables)
        
        # Select samples to maximize minimum distance
        selected_indices = self._maximin_selection(numeric_candidates, n_samples, max_iterations)
        
        return candidates.iloc[selected_indices].reset_index(drop=True)
    
    def _convert_to_numeric(self, data: pd.DataFrame, variables: List[Variable]) -> np.ndarray:
        """Convert categorical variables to numeric for distance calculation."""
        numeric_data = np.zeros((len(data), len(variables)))
        
        for i, var in enumerate(variables):
            if var.type == VariableType.CATEGORICAL:
                # Convert categories to numeric codes
                unique_cats = var.categories
                cat_to_num = {cat: j for j, cat in enumerate(unique_cats)}
                numeric_data[:, i] = [cat_to_num[val] for val in data[var.name]]
            else:
                numeric_data[:, i] = data[var.name].astype(float)
        
        # Normalize to [0, 1] for fair distance calculations
        scaler = MinMaxScaler()
        return scaler.fit_transform(numeric_data)
    
    def _maximin_selection(self, candidates: np.ndarray, n_samples: int, 
                          max_iterations: int) -> List[int]:
        """Select samples to maximize minimum pairwise distance."""
        n_candidates = len(candidates)
        
        if n_samples >= n_candidates:
            return list(range(n_candidates))
        
        # Start with random selection
        rng = np.random.RandomState(42)
        selected = list(rng.choice(n_candidates, n_samples, replace=False))
        
        best_min_dist = self._calculate_min_distance(candidates[selected])
        
        # Iterative improvement
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(n_samples):
                # Try replacing selected[i] with each unselected candidate
                current_selected = selected.copy()
                unselected = [j for j in range(n_candidates) if j not in selected]
                
                for candidate_idx in unselected:
                    test_selected = current_selected.copy()
                    test_selected[i] = candidate_idx
                    
                    test_min_dist = self._calculate_min_distance(candidates[test_selected])
                    
                    if test_min_dist > best_min_dist:
                        selected = test_selected
                        best_min_dist = test_min_dist
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return selected
    
    def _calculate_min_distance(self, points: np.ndarray) -> float:
        """Calculate minimum pairwise distance."""
        if len(points) < 2:
            return float('inf')
        
        distances = pdist(points)
        return np.min(distances)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Maximin-specific parameters."""
        return {
            'max_iterations': {
                'type': 'integer',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Maximum optimization iterations'
            },
            'n_candidates': {
                'type': 'integer',
                'default': None,  # Will be set to n_samples * 10
                'min': 100,
                'max': 10000,
                'description': 'Number of candidate samples to generate'
            }
        }


class SamplingEngine:
    """Main engine for managing sampling algorithms and quality metrics."""
    
    def __init__(self):
        """Initialize the sampling engine with all available algorithms."""
        self.algorithms = {
            "Latin Hypercube Sampling": LatinHypercubeSampling(),
            "Sobol Sequences": SobolSampling(),
            "Halton Sequences": HaltonSampling(),
            "Random Sampling": RandomSampling(),
            "Uniform Grid Sampling": UniformGridSampling(),
            "Orthogonal Arrays": OrthogonalArraySampling(),
            "Maximin Distance Design": MaximinDistanceSampling()
        }
        
        # Filter out unavailable algorithms
        self.available_algorithms = {
            name: alg for name, alg in self.algorithms.items() 
            if alg.is_available()
        }
        
        self.validator = ValidationUtils()
    
    def get_available_algorithms(self) -> Dict[str, str]:
        """Get list of available algorithms with descriptions."""
        return {name: alg.description for name, alg in self.available_algorithms.items()}
    
    def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """Get parameters for a specific algorithm."""
        if algorithm_name not in self.available_algorithms:
            return {}
        
        return self.available_algorithms[algorithm_name].get_parameters()
    
    def generate_samples(self, variables: List[Variable], algorithm: str, 
                        n_samples: int, random_state: Optional[int] = None,
                        **algorithm_params) -> pd.DataFrame:
        """Generate samples using specified algorithm."""
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Algorithm '{algorithm}' is not available")
        
        if not variables:
            raise ValueError("No variables defined")
        
        if n_samples < 1:
            raise ValueError("Number of samples must be positive")
        
        # Validate minimum samples
        min_samples = max(len(variables) ** 2, 4)
        if n_samples < min_samples:
            warnings.warn(f"Recommended minimum samples: {min_samples} for {len(variables)} variables")
        
        # Generate samples
        sampler = self.available_algorithms[algorithm]
        samples = sampler.generate(variables, n_samples, random_state, **algorithm_params)
        
        # Add experiment index
        samples.insert(0, 'Experiment_ID', range(1, len(samples) + 1))
        
        return samples
    
    def calculate_quality_metrics(self, samples: pd.DataFrame, 
                                variables: List[Variable]) -> Dict[str, Any]:
        """Calculate quality metrics for generated samples."""
        if samples.empty:
            return {}
        
        # Remove experiment ID column for calculations
        data_cols = [var.name for var in variables]
        data = samples[data_cols]
        
        metrics = {}
        
        # Basic metrics
        metrics['sample_count'] = len(samples)
        metrics['variable_count'] = len(variables)
        
        # Space-filling metrics
        metrics.update(self._calculate_space_filling_metrics(data, variables))
        
        # Statistical properties
        metrics.update(self._calculate_statistical_metrics(data, variables))
        
        # Coverage metrics
        metrics.update(self._calculate_coverage_metrics(data, variables))
        
        return metrics
    
    def _calculate_space_filling_metrics(self, data: pd.DataFrame, 
                                       variables: List[Variable]) -> Dict[str, Any]:
        """Calculate space-filling quality metrics."""
        metrics = {}
        
        # Convert to numeric for distance calculations
        numeric_data = self._convert_to_numeric_for_metrics(data, variables)
        
        if numeric_data.size > 0:
            # Minimum distance
            if len(data) > 1:
                distances = pdist(numeric_data)
                metrics['min_distance'] = float(np.min(distances))
                metrics['mean_distance'] = float(np.mean(distances))
                metrics['max_distance'] = float(np.max(distances))
                
                # Distance uniformity (coefficient of variation)
                metrics['distance_uniformity'] = float(np.std(distances) / np.mean(distances))
            
            # Discrepancy (simplified calculation)
            metrics['star_discrepancy'] = self._calculate_star_discrepancy(numeric_data)
        
        return metrics
    
    def _calculate_statistical_metrics(self, data: pd.DataFrame, 
                                     variables: List[Variable]) -> Dict[str, Any]:
        """Calculate statistical quality metrics."""
        metrics = {}
        
        # Variable-wise statistics
        variable_stats = {}
        correlation_data = []
        
        for var in variables:
            if var.name in data.columns:
                col_data = data[var.name]
                
                if var.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    correlation_data.append(numeric_data)
                    
                    var_stats = {
                        'mean': float(numeric_data.mean()),
                        'std': float(numeric_data.std()),
                        'min': float(numeric_data.min()),
                        'max': float(numeric_data.max()),
                        'range_coverage': self._calculate_range_coverage(numeric_data, var)
                    }
                
                elif var.type == VariableType.CATEGORICAL:
                    value_counts = col_data.value_counts()
                    var_stats = {
                        'unique_categories': len(value_counts),
                        'category_distribution': value_counts.to_dict(),
                        'uniformity': self._calculate_categorical_uniformity(value_counts)
                    }
                
                variable_stats[var.name] = var_stats
        
        metrics['variable_statistics'] = variable_stats
        
        # Correlation analysis for numeric variables
        if len(correlation_data) > 1:
            corr_matrix = np.corrcoef(correlation_data)
            
            # Extract upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            metrics['correlation_stats'] = {
                'max_correlation': float(np.max(np.abs(upper_tri))),
                'mean_correlation': float(np.mean(np.abs(upper_tri))),
                'correlation_matrix': corr_matrix.tolist()
            }
        
        return metrics
    
    def _calculate_coverage_metrics(self, data: pd.DataFrame, 
                                  variables: List[Variable]) -> Dict[str, Any]:
        """Calculate coverage quality metrics."""
        metrics = {}
        
        total_coverage = 1.0
        coverage_details = {}
        
        for var in variables:
            if var.name not in data.columns:
                continue
            
            col_data = data[var.name]
            coverage = 1.0  # Initialize coverage variable
            
            if var.type == VariableType.CONTINUOUS:
                # Calculate range coverage
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                data_min = numeric_data.min()
                data_max = numeric_data.max()
                data_range = data_max - data_min
                total_range = var.max_value - var.min_value
                coverage = (data_range / total_range) if total_range > 0 else 1.0
                
            elif var.type == VariableType.DISCRETE:
                # Calculate discrete value coverage
                possible_values = set(var.get_discrete_values())
                covered_values = set(col_data.unique())
                coverage = len(covered_values) / len(possible_values)
                
            elif var.type == VariableType.CATEGORICAL:
                # Calculate category coverage
                covered_categories = set(col_data.unique())
                total_categories = set(var.categories)
                coverage = len(covered_categories) / len(total_categories)
            
            coverage_details[var.name] = float(coverage)
            total_coverage *= coverage
        
        metrics['coverage_per_variable'] = coverage_details
        metrics['overall_coverage'] = float(total_coverage)
        
        return metrics
    
    def _convert_to_numeric_for_metrics(self, data: pd.DataFrame, 
                                      variables: List[Variable]) -> np.ndarray:
        """Convert data to numeric format for metric calculations."""
        numeric_data = np.zeros((len(data), len(variables)))
        
        for i, var in enumerate(variables):
            if var.name not in data.columns:
                continue
            
            col_data = data[var.name]
            
            if var.type == VariableType.CATEGORICAL:
                # Convert categories to numeric codes
                unique_vals = col_data.unique()
                val_to_num = {val: j for j, val in enumerate(unique_vals)}
                numeric_data[:, i] = [val_to_num[val] for val in col_data]
            else:
                numeric_data[:, i] = pd.to_numeric(col_data, errors='coerce')
        
        # Normalize to [0, 1] for consistent metrics
        for i in range(numeric_data.shape[1]):
            col_min, col_max = numeric_data[:, i].min(), numeric_data[:, i].max()
            if col_max > col_min:
                numeric_data[:, i] = (numeric_data[:, i] - col_min) / (col_max - col_min)
        
        return numeric_data
    
    def _calculate_star_discrepancy(self, points: np.ndarray) -> float:
        """Calculate star discrepancy (simplified version)."""
        if points.size == 0 or len(points) < 2:
            return 0.0
        
        n_points, n_dims = points.shape
        
        # Simplified discrepancy calculation
        # For each point, calculate the volume of the box from origin
        volumes = np.prod(points, axis=1)
        empirical_measure = np.mean(volumes)
        
        # Theoretical measure for uniform distribution
        theoretical_measure = 1.0 / (2 ** n_dims)
        
        return abs(empirical_measure - theoretical_measure)
    
    def _calculate_range_coverage(self, data: pd.Series, variable: Variable) -> float:
        """Calculate how well the data covers the variable's range."""
        if variable.type not in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
            return 1.0
        
        data_min, data_max = data.min(), data.max()
        var_min, var_max = variable.min_value, variable.max_value
        
        if var_max == var_min:
            return 1.0
        
        return (data_max - data_min) / (var_max - var_min)
    
    def _calculate_categorical_uniformity(self, value_counts: pd.Series) -> float:
        """Calculate uniformity of categorical distribution."""
        if len(value_counts) <= 1:
            return 1.0
        
        # Calculate entropy-based uniformity
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(len(value_counts))
        
        return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def compare_algorithms(self, variables: List[Variable], n_samples: int,
                          algorithms: List[str], random_state: Optional[int] = None) -> pd.DataFrame:
        """Compare multiple algorithms on the same problem."""
        results = []
        
        for algorithm in algorithms:
            if algorithm not in self.available_algorithms:
                continue
            
            try:
                # Generate samples
                samples = self.generate_samples(variables, algorithm, n_samples, random_state)
                
                # Calculate metrics
                metrics = self.calculate_quality_metrics(samples, variables)
                
                # Add algorithm name and compile results
                result = {'Algorithm': algorithm}
                result.update(metrics)
                results.append(result)
                
            except Exception as e:
                # Add failed algorithm with error info
                results.append({
                    'Algorithm': algorithm,
                    'Error': str(e)
                })
        
        return pd.DataFrame(results)